library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Conv3 VALID: 30x30x32 -> Conv3x3, ReLU -> 28x28x64 (NO pooling: MaxPool3 runs on the ARM HPS).
-- Eight parallel 9x8 MAC lanes, shared 32x64 Q40 requantizer.
-- ROMs: 18432x8 weights, 64x32 biases (256 bytes).
-- Input and output Avalon addresses are BYTE addresses (same convention as Conv1/Conv2).
entity conv3_finale is
    port (
        clk : in std_logic;
        reset : in std_logic;
        avs_address : in std_logic_vector(2 downto 0);
        avs_write, avs_read : in std_logic;
        avs_writedata : in std_logic_vector(31 downto 0);
        avs_readdata : out std_logic_vector(31 downto 0);
        avs_waitrequest : out std_logic;
        avm_address : out std_logic_vector(15 downto 0);
        avm_read : out std_logic;
        avm_readdata : in std_logic_vector(31 downto 0);
        avm_waitrequest, avm_readdatavalid : in std_logic;
        weight_address : out std_logic_vector(14 downto 0);
        weight_read : out std_logic;
        weight_readdata : in std_logic_vector(7 downto 0);
        weight_waitrequest, weight_readdatavalid : in std_logic;
        bias_address : out std_logic_vector(7 downto 0);
        bias_read : out std_logic;
        bias_readdata : in std_logic_vector(31 downto 0);
        bias_waitrequest, bias_readdatavalid : in std_logic;
        output_address : out std_logic_vector(15 downto 0);
        output_write : out std_logic;
        output_writedata : out std_logic_vector(31 downto 0);
        output_byteenable : out std_logic_vector(3 downto 0);
        output_waitrequest : in std_logic
    );
end entity;

architecture rtl of conv3_finale is
    -- 8 parallel MACs, one pipelined tap per clock.
    -- 8 synchronous 2304x8 weight banks to avoid a 18432-to-1 LUT mux.
    -- 3 x 1024-byte input rows (correct for 32 pixels x 32 channels).
    type state_t is (IDLE,W_REQ,W_WAIT,B_REQ,B_WAIT,ROW_PREP,ROW_REQ,
                     ROW_WAIT,TAP_INIT,TAP_RUN,TAP_DRAIN,TAP_DRAIN2,POOL,
                     QUANT,WRITE_WORD,ADVANCE,FINISHED);
    signal state : state_t := IDLE;
    type rows_t is array(0 to 719) of std_logic_vector(31 downto 0);
    signal rows : rows_t;
    signal row_rd_addr : integer range 0 to 719 := 0;
    signal row_rd_data : std_logic_vector(31 downto 0) := (others=>'0');
    signal row_wr_en : std_logic := '0';
    signal row_wr_addr : integer range 0 to 767 := 0;
    signal row_wr_data : std_logic_vector(31 downto 0) := (others=>'0');

    type bank_t is array (0 to 2303) of signed(7 downto 0);
    signal wb0,wb1,wb2,wb3,wb4,wb5,wb6,wb7 : bank_t;
    attribute ramstyle : string;
    attribute ramstyle of wb0 : signal is "M10K";
    attribute ramstyle of wb1 : signal is "M10K";
    attribute ramstyle of wb2 : signal is "M10K";
    attribute ramstyle of wb3 : signal is "M10K";
    attribute ramstyle of wb4 : signal is "M10K";
    attribute ramstyle of wb5 : signal is "M10K";
    attribute ramstyle of wb6 : signal is "M10K";
    attribute ramstyle of wb7 : signal is "M10K";

    signal wr_bank : integer range 0 to 7 := 0;
    signal wr_addr : integer range 0 to 2303 := 0;
    signal wr_data : signed(7 downto 0) := (others=>'0');
    signal wr_en : std_logic := '0';
    signal weight_rd_addr : integer range 0 to 2303 := 0;
    signal wd0,wd1,wd2,wd3,wd4,wd5,wd6,wd7 : signed(7 downto 0) := (others=>'0');

    type biases_t is array (0 to 63) of signed(31 downto 0);
    signal biases : biases_t;
    type lanes_t is array (0 to 7) of signed(31 downto 0);
    signal acc, pooled : lanes_t := (others=>(others=>'0'));
    signal widx : integer range 0 to 18431 := 0;
    signal bidx : integer range 0 to 63 := 0;
    signal row_slot : integer range 0 to 3 := 0;
    signal row_word : integer range 0 to 239 := 0;
    signal py, px : integer range 0 to 27 := 0;
    signal grp : integer range 0 to 7 := 0;
    signal sub : integer range 0 to 3 := 0;
    signal tap_idx : integer range 0 to 287 := 0;
    signal quant_lane : integer range 0 to 7 := 0;
    signal pipe_valid : std_logic := '0';
    signal issue_valid : std_logic := '0';
    signal issue_first : std_logic := '0';
    signal issue_pixel_valid : std_logic := '0';
    signal issue_byte_sel : integer range 0 to 3 := 0;
    signal pipe_first : std_logic := '0';
    signal pipe_pixel_valid : std_logic := '0';
    signal pipe_byte_sel : integer range 0 to 3 := 0;
    signal image_addr_reg : std_logic_vector(15 downto 0) := (others=>'0');
    signal out_word : std_logic_vector(63 downto 0) := (others=>'0');
    signal completed : integer range 0 to 50176 := 0;
    signal busy, done : std_logic := '0';
    signal dbg_image,dbg_weight,dbg_bias : std_logic_vector(31 downto 0) := (others=>'0');

    function quantize(acc_in : signed(31 downto 0); f : integer) return std_logic_vector is
        variable m : signed(31 downto 0);
        variable p : signed(63 downto 0);
        variable mag, rounded : unsigned(63 downto 0);
        variable q : integer;
    begin
        case f is
            when 0 => m := to_signed(998661445,32);
            when 1 => m := to_signed(838849766,32);
            when 2 => m := to_signed(708527931,32);
            when 3 => m := to_signed(814961541,32);
            when 4 => m := to_signed(1024716646,32);
            when 5 => m := to_signed(631356120,32);
            when 6 => m := to_signed(1029098312,32);
            when 7 => m := to_signed(785780674,32);
            when 8 => m := to_signed(1677666991,32);
            when 9 => m := to_signed(1039800157,32);
            when 10 => m := to_signed(953221948,32);
            when 11 => m := to_signed(1070272599,32);
            when 12 => m := to_signed(977891824,32);
            when 13 => m := to_signed(760329565,32);
            when 14 => m := to_signed(1079446820,32);
            when 15 => m := to_signed(837528371,32);
            when 16 => m := to_signed(329542846,32);
            when 17 => m := to_signed(575809348,32);
            when 18 => m := to_signed(874180576,32);
            when 19 => m := to_signed(784201519,32);
            when 20 => m := to_signed(476853720,32);
            when 21 => m := to_signed(842447425,32);
            when 22 => m := to_signed(576944603,32);
            when 23 => m := to_signed(528532777,32);
            when 24 => m := to_signed(685762406,32);
            when 25 => m := to_signed(712323502,32);
            when 26 => m := to_signed(1011774634,32);
            when 27 => m := to_signed(332077002,32);
            when 28 => m := to_signed(1344178349,32);
            when 29 => m := to_signed(1167374465,32);
            when 30 => m := to_signed(1751784809,32);
            when 31 => m := to_signed(835162489,32);
            when 32 => m := to_signed(1636312319,32);
            when 33 => m := to_signed(1049383279,32);
            when 34 => m := to_signed(898884039,32);
            when 35 => m := to_signed(616362959,32);
            when 36 => m := to_signed(758135436,32);
            when 37 => m := to_signed(943113051,32);
            when 38 => m := to_signed(761257963,32);
            when 39 => m := to_signed(796866413,32);
            when 40 => m := to_signed(1079040430,32);
            when 41 => m := to_signed(313851058,32);
            when 42 => m := to_signed(323986034,32);
            when 43 => m := to_signed(840866857,32);
            when 44 => m := to_signed(726094930,32);
            when 45 => m := to_signed(788568536,32);
            when 46 => m := to_signed(1020553870,32);
            when 47 => m := to_signed(933446119,32);
            when 48 => m := to_signed(1025388382,32);
            when 49 => m := to_signed(1234436438,32);
            when 50 => m := to_signed(491735239,32);
            when 51 => m := to_signed(931398475,32);
            when 52 => m := to_signed(852429299,32);
            when 53 => m := to_signed(454539210,32);
            when 54 => m := to_signed(399379680,32);
            when 55 => m := to_signed(1161856184,32);
            when 56 => m := to_signed(768169312,32);
            when 57 => m := to_signed(848562631,32);
            when 58 => m := to_signed(1141520157,32);
            when 59 => m := to_signed(709082218,32);
            when 60 => m := to_signed(780517276,32);
            when 61 => m := to_signed(891596120,32);
            when 62 => m := to_signed(726164405,32);
            when 63 => m := to_signed(1646785125,32);
            when others => m := (others=>'0');
        end case;
        p := acc_in*m;
        if p(63)='1' then mag:=unsigned(-p); else mag:=unsigned(p); end if;
        rounded := shift_right(mag + shift_left(to_unsigned(1,64),39),40);
        if rounded > to_unsigned(1000000,64) then
            if p(63)='1' then q:=-128; else q:=127; end if;
        else
            q:=to_integer(rounded(31 downto 0));
            if p(63)='1' then q:=-q; end if;
            q:=q-128;
            if q < -128 then q:=-128; end if;
            if q > 127 then q:=127; end if;
        end if;
        -- Fused ReLU with output zero point -128: saturation already applies.
        return std_logic_vector(to_signed(q,8));
    end function;
begin
    avs_waitrequest <= '0';
    avm_address <= image_addr_reg;
    avm_read <= '1' when state=ROW_REQ else '0';
    weight_address <= std_logic_vector(to_unsigned(widx,15));
    weight_read <= '1' when state=W_REQ else '0';
    bias_address <= std_logic_vector(to_unsigned(bidx*4,8));
    bias_read <= '1' when state=B_REQ else '0';
    output_address <= std_logic_vector(to_unsigned((py*28+px)*64+grp*8+sub*4,16));
    output_writedata <= out_word(31 downto 0) when sub=0 else out_word(63 downto 32);
    output_byteenable <= "1111";
    output_write <= '1' when state=WRITE_WORD else '0';

    -- Synchronous line-buffer RAM (one read port, one write port).
    process(clk)
    begin
        if rising_edge(clk) then
            row_rd_data <= rows(row_rd_addr);
            if row_wr_en='1' then rows(row_wr_addr)<=row_wr_data; end if;
        end if;
    end process;

    -- Eight independent synchronous banks. Each bank has one read and
    -- one write port, and can infer block RAM instead of wide mux logic.
    process(clk)
    begin
        if rising_edge(clk) then
            wd0<=wb0(weight_rd_addr);
            wd1<=wb1(weight_rd_addr);
            wd2<=wb2(weight_rd_addr);
            wd3<=wb3(weight_rd_addr);
            wd4<=wb4(weight_rd_addr);
            wd5<=wb5(weight_rd_addr);
            wd6<=wb6(weight_rd_addr);
            wd7<=wb7(weight_rd_addr);
            if wr_en='1' then
                case wr_bank is
                    when 0 => wb0(wr_addr)<=wr_data;
                    when 1 => wb1(wr_addr)<=wr_data;
                    when 2 => wb2(wr_addr)<=wr_data;
                    when 3 => wb3(wr_addr)<=wr_data;
                    when 4 => wb4(wr_addr)<=wr_data;
                    when 5 => wb5(wr_addr)<=wr_data;
                    when 6 => wb6(wr_addr)<=wr_data;
                    when others => wb7(wr_addr)<=wr_data;
                end case;
            end if;
        end if;
    end process;

    process(clk,reset)
        variable iy,ix,ky,kx,ch,byte_addr,slot : integer;
        variable bank, local_idx, pixel_i : integer;
        variable product_v : signed(16 downto 0);
        variable pix : signed(8 downto 0);
        variable packed : std_logic_vector(63 downto 0);
    begin
        if reset='1' then
            state<=IDLE; widx<=0; bidx<=0; row_slot<=0; row_word<=0;
            py<=0; px<=0; grp<=0; sub<=0; tap_idx<=0; quant_lane<=0;
            completed<=0; busy<='0'; done<='0';
            image_addr_reg<=(others=>'0'); out_word<=(others=>'0');
            dbg_image<=(others=>'0'); dbg_weight<=(others=>'0'); dbg_bias<=(others=>'0');
            acc<=(others=>(others=>'0')); pooled<=(others=>(others=>'0'));
            row_rd_addr<=0; row_wr_addr<=0; row_wr_en<='0';
            row_wr_data<=(others=>'0'); pipe_valid<='0';
            pipe_first<='0'; pipe_pixel_valid<='0'; pipe_byte_sel<=0;
            issue_valid<='0'; issue_first<='0'; issue_pixel_valid<='0'; issue_byte_sel<=0;
            wr_en<='0'; wr_bank<=0; wr_addr<=0; wr_data<=(others=>'0');
            weight_rd_addr<=0;
        elsif rising_edge(clk) then
            row_wr_en<='0'; wr_en<='0';
            case state is
                when IDLE =>
                    busy<='0';
                    if avs_write='1' and avs_address="000" and avs_writedata(0)='1' then
                        busy<='1'; done<='0'; completed<=0;
                        widx<=0; bidx<=0; row_slot<=0; row_word<=0;
                        py<=0; px<=0; grp<=0; sub<=0; tap_idx<=0;
                        pipe_valid<='0'; state<=W_REQ;
                    end if;
                when W_REQ =>
                    if weight_waitrequest='0' then
                        if weight_readdatavalid='1' then
                            local_idx:=widx mod 288;
                            bank:=(widx/288) mod 8;
                            wr_bank<=bank;
                            wr_addr<=((widx/288)/8)*288+local_idx;
                            wr_data<=signed(weight_readdata); wr_en<='1';
                            if widx=0 then dbg_weight<=std_logic_vector(resize(signed(weight_readdata),32)); end if;
                            if widx=18431 then state<=B_REQ; else widx<=widx+1; end if;
                        else state<=W_WAIT; end if;
                    end if;
                when W_WAIT =>
                    if weight_readdatavalid='1' then
                        local_idx:=widx mod 288;
                        bank:=(widx/288) mod 8;
                        wr_bank<=bank;
                        wr_addr<=((widx/288)/8)*288+local_idx;
                        wr_data<=signed(weight_readdata); wr_en<='1';
                        if widx=0 then dbg_weight<=std_logic_vector(resize(signed(weight_readdata),32)); end if;
                        if widx=18431 then state<=B_REQ; else widx<=widx+1; state<=W_REQ; end if;
                    end if;
                when B_REQ =>
                    if bias_waitrequest='0' then
                        if bias_readdatavalid='1' then
                            biases(bidx)<=signed(bias_readdata);
                            if bidx=0 then dbg_bias<=bias_readdata; end if;
                            if bidx=63 then state<=ROW_PREP; else bidx<=bidx+1; end if;
                        else state<=B_WAIT; end if;
                    end if;
                when B_WAIT =>
                    if bias_readdatavalid='1' then
                        biases(bidx)<=signed(bias_readdata);
                        if bidx=0 then dbg_bias<=bias_readdata; end if;
                        if bidx=63 then state<=ROW_PREP; else bidx<=bidx+1; state<=B_REQ; end if;
                    end if;
                when ROW_PREP =>
                    if row_slot=3 then
                        px<=0; grp<=0; sub<=0; tap_idx<=0; state<=TAP_INIT;
                    else
                        iy:=py+row_slot;
                        if iy<0 or iy>=30 then
                            row_word<=0; row_slot<=row_slot+1;
                        else
                            image_addr_reg<=std_logic_vector(to_unsigned(iy*960+row_word*4,image_addr_reg'length));
                            state<=ROW_REQ;
                        end if;
                    end if;
                when ROW_REQ =>
                    if avm_waitrequest='0' then
                        if avm_readdatavalid='1' then
                            row_wr_addr<=row_slot*240+row_word;
                            row_wr_data<=avm_readdata; row_wr_en<='1';
                            if row_slot=1 and row_word=0 then dbg_image<=avm_readdata; end if;
                            if row_word=239 then row_word<=0; row_slot<=row_slot+1;
                            else row_word<=row_word+1; end if;
                            state<=ROW_PREP;
                        else state<=ROW_WAIT; end if;
                    end if;
                when ROW_WAIT =>
                    if avm_readdatavalid='1' then
                        row_wr_addr<=row_slot*240+row_word;
                        row_wr_data<=avm_readdata; row_wr_en<='1';
                        if row_slot=1 and row_word=0 then dbg_image<=avm_readdata; end if;
                        if row_word=239 then row_word<=0; row_slot<=row_slot+1;
                        else row_word<=row_word+1; end if;
                        state<=ROW_PREP;
                    end if;
                when TAP_INIT =>
                    pipe_valid<='0'; tap_idx<=0; state<=TAP_RUN;
                when TAP_RUN =>
                    -- Retire the previously issued tap while issuing the next.
                    if pipe_valid='1' then
                        if pipe_pixel_valid='1' then
                            pixel_i:=to_integer(signed(row_rd_data(pipe_byte_sel*8+7 downto pipe_byte_sel*8)))+128;
                        else pixel_i:=0; end if;
                        pix:=to_signed(pixel_i,9);
                        product_v:=pix*wd0;
                        if pipe_first='1' then acc(0)<=biases(grp*8+0)+resize(product_v,32);
                        else acc(0)<=acc(0)+resize(product_v,32); end if;
                        product_v:=pix*wd1;
                        if pipe_first='1' then acc(1)<=biases(grp*8+1)+resize(product_v,32);
                        else acc(1)<=acc(1)+resize(product_v,32); end if;
                        product_v:=pix*wd2;
                        if pipe_first='1' then acc(2)<=biases(grp*8+2)+resize(product_v,32);
                        else acc(2)<=acc(2)+resize(product_v,32); end if;
                        product_v:=pix*wd3;
                        if pipe_first='1' then acc(3)<=biases(grp*8+3)+resize(product_v,32);
                        else acc(3)<=acc(3)+resize(product_v,32); end if;
                        product_v:=pix*wd4;
                        if pipe_first='1' then acc(4)<=biases(grp*8+4)+resize(product_v,32);
                        else acc(4)<=acc(4)+resize(product_v,32); end if;
                        product_v:=pix*wd5;
                        if pipe_first='1' then acc(5)<=biases(grp*8+5)+resize(product_v,32);
                        else acc(5)<=acc(5)+resize(product_v,32); end if;
                        product_v:=pix*wd6;
                        if pipe_first='1' then acc(6)<=biases(grp*8+6)+resize(product_v,32);
                        else acc(6)<=acc(6)+resize(product_v,32); end if;
                        product_v:=pix*wd7;
                        if pipe_first='1' then acc(7)<=biases(grp*8+7)+resize(product_v,32);
                        else acc(7)<=acc(7)+resize(product_v,32); end if;
                    end if;
                    ky:=tap_idx/96; kx:=(tap_idx mod 96)/32; ch:=tap_idx mod 32;
                    ix:=px+kx;
                    iy:=py+ky;
                    if ix<0 or ix>=30 or iy<0 or iy>=30 then
                        row_rd_addr<=0; issue_byte_sel<=0; issue_pixel_valid<='0';
                    else
                        slot:=iy-py;
                        byte_addr:=ix*32+ch;
                        row_rd_addr<=slot*240+byte_addr/4;
                        issue_byte_sel<=byte_addr mod 4;
                        issue_pixel_valid<='1';
                    end if;
                    weight_rd_addr<=grp*288+tap_idx;
                    pipe_valid<=issue_valid; pipe_first<=issue_first;
                    pipe_pixel_valid<=issue_pixel_valid; pipe_byte_sel<=issue_byte_sel;
                    issue_valid<='1';
                    if tap_idx=0 then issue_first<='1'; else issue_first<='0'; end if;
                    if tap_idx=287 then state<=TAP_DRAIN;
                    else tap_idx<=tap_idx+1; end if;
                when TAP_DRAIN =>
                    if pipe_valid='1' then
                        if pipe_pixel_valid='1' then
                            pixel_i:=to_integer(signed(row_rd_data(pipe_byte_sel*8+7 downto pipe_byte_sel*8)))+128;
                        else pixel_i:=0; end if;
                        pix:=to_signed(pixel_i,9);
                        product_v:=pix*wd0;
                        if pipe_first='1' then acc(0)<=biases(grp*8+0)+resize(product_v,32);
                        else acc(0)<=acc(0)+resize(product_v,32); end if;
                        product_v:=pix*wd1;
                        if pipe_first='1' then acc(1)<=biases(grp*8+1)+resize(product_v,32);
                        else acc(1)<=acc(1)+resize(product_v,32); end if;
                        product_v:=pix*wd2;
                        if pipe_first='1' then acc(2)<=biases(grp*8+2)+resize(product_v,32);
                        else acc(2)<=acc(2)+resize(product_v,32); end if;
                        product_v:=pix*wd3;
                        if pipe_first='1' then acc(3)<=biases(grp*8+3)+resize(product_v,32);
                        else acc(3)<=acc(3)+resize(product_v,32); end if;
                        product_v:=pix*wd4;
                        if pipe_first='1' then acc(4)<=biases(grp*8+4)+resize(product_v,32);
                        else acc(4)<=acc(4)+resize(product_v,32); end if;
                        product_v:=pix*wd5;
                        if pipe_first='1' then acc(5)<=biases(grp*8+5)+resize(product_v,32);
                        else acc(5)<=acc(5)+resize(product_v,32); end if;
                        product_v:=pix*wd6;
                        if pipe_first='1' then acc(6)<=biases(grp*8+6)+resize(product_v,32);
                        else acc(6)<=acc(6)+resize(product_v,32); end if;
                        product_v:=pix*wd7;
                        if pipe_first='1' then acc(7)<=biases(grp*8+7)+resize(product_v,32);
                        else acc(7)<=acc(7)+resize(product_v,32); end if;
                    end if;
                    pipe_valid<=issue_valid; pipe_first<=issue_first;
                    pipe_pixel_valid<=issue_pixel_valid; pipe_byte_sel<=issue_byte_sel;
                    issue_valid<='0'; state<=TAP_DRAIN2;
                when TAP_DRAIN2 =>
                    if pipe_valid='1' then
                        if pipe_pixel_valid='1' then
                            pixel_i:=to_integer(signed(row_rd_data(pipe_byte_sel*8+7 downto pipe_byte_sel*8)))+128;
                        else pixel_i:=0; end if;
                        pix:=to_signed(pixel_i,9);
                        product_v:=pix*wd0;
                        if pipe_first='1' then acc(0)<=biases(grp*8+0)+resize(product_v,32);
                        else acc(0)<=acc(0)+resize(product_v,32); end if;
                        product_v:=pix*wd1;
                        if pipe_first='1' then acc(1)<=biases(grp*8+1)+resize(product_v,32);
                        else acc(1)<=acc(1)+resize(product_v,32); end if;
                        product_v:=pix*wd2;
                        if pipe_first='1' then acc(2)<=biases(grp*8+2)+resize(product_v,32);
                        else acc(2)<=acc(2)+resize(product_v,32); end if;
                        product_v:=pix*wd3;
                        if pipe_first='1' then acc(3)<=biases(grp*8+3)+resize(product_v,32);
                        else acc(3)<=acc(3)+resize(product_v,32); end if;
                        product_v:=pix*wd4;
                        if pipe_first='1' then acc(4)<=biases(grp*8+4)+resize(product_v,32);
                        else acc(4)<=acc(4)+resize(product_v,32); end if;
                        product_v:=pix*wd5;
                        if pipe_first='1' then acc(5)<=biases(grp*8+5)+resize(product_v,32);
                        else acc(5)<=acc(5)+resize(product_v,32); end if;
                        product_v:=pix*wd6;
                        if pipe_first='1' then acc(6)<=biases(grp*8+6)+resize(product_v,32);
                        else acc(6)<=acc(6)+resize(product_v,32); end if;
                        product_v:=pix*wd7;
                        if pipe_first='1' then acc(7)<=biases(grp*8+7)+resize(product_v,32);
                        else acc(7)<=acc(7)+resize(product_v,32); end if;
                    end if;
                    pipe_valid<='0'; state<=POOL;
                when POOL =>
                    for l in 0 to 7 loop
                        pooled(l)<=acc(l);
                    end loop;
                    quant_lane<=0; out_word<=(others=>'0'); state<=QUANT;
                when QUANT =>
                    packed:=out_word;
                    packed(quant_lane*8+7 downto quant_lane*8):=quantize(pooled(quant_lane),grp*8+quant_lane);
                    out_word<=packed;
                    if quant_lane=7 then state<=WRITE_WORD;
                    else quant_lane<=quant_lane+1; end if;
                when WRITE_WORD =>
                    if output_waitrequest='0' then
                        if sub=0 then sub<=1; else completed<=completed+8; state<=ADVANCE; end if;
                    end if;
                when ADVANCE =>
                    sub<=0; tap_idx<=0;
                    if grp=7 then
                        grp<=0;
                        if px=27 then
                            px<=0;
                            if py=27 then state<=FINISHED;
                            else py<=py+1; row_slot<=0; row_word<=0; state<=ROW_PREP; end if;
                        else px<=px+1; state<=TAP_INIT; end if;
                    else grp<=grp+1; state<=TAP_INIT; end if;
                when FINISHED =>
                    busy<='0'; done<='1'; state<=IDLE;
            end case;
        end if;
    end process;

    process(avs_address,avs_read,busy,done,dbg_image,dbg_weight,dbg_bias,acc,px,py,grp,completed)
    begin
        avs_readdata<=(others=>'0');
        if avs_read='1' then
            case avs_address is
                when "001" => avs_readdata(0)<=busy; avs_readdata(1)<=done;
                when "010" => avs_readdata<=dbg_image;
                when "011" => avs_readdata<=dbg_weight;
                when "100" => avs_readdata<=dbg_bias;
                when "101" => avs_readdata<=std_logic_vector(acc(0));
                when "110" =>
                    avs_readdata(5 downto 0)<=std_logic_vector(to_unsigned(px,6));
                    avs_readdata(13 downto 8)<=std_logic_vector(to_unsigned(py,6));
                    avs_readdata(22 downto 16)<=std_logic_vector(to_unsigned(grp*8,7));
                when "111" => avs_readdata<=std_logic_vector(to_unsigned(completed,32));
                when others => null;
            end case;
        end if;
    end process;
end architecture;
