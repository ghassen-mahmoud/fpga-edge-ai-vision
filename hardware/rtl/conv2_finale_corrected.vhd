library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Conv2 VALID: 63x63x16 -> 61x61x32 -> MaxPool 30x30x32.
-- Eight parallel output-channel MACs. Four-row on-chip buffer.
-- Avalon-MM master addresses are BYTE addresses, assuming byte-addressed slaves.
-- Weights: OHWI, 32*3*3*16 = 4608 INT8, packed 8 bytes per 64-bit word.
-- Biases: 32 INT32. Output: 32768 bytes, packed 4 channels per 32-bit word.
-- MULT/SHIFT: per-channel requantization constants extracted from bottle_cnn_int8.tflite.
entity conv2_finale is
  port (
    clk, reset : in std_logic;
    avs_address : in std_logic_vector(2 downto 0);
    avs_read, avs_write : in std_logic;
    avs_writedata : in std_logic_vector(31 downto 0);
    avs_readdata : out std_logic_vector(31 downto 0);
    avs_waitrequest : out std_logic;
    input_address : out std_logic_vector(15 downto 0);
    input_read : out std_logic;
    input_readdata : in std_logic_vector(31 downto 0);
    input_waitrequest, input_readdatavalid : in std_logic;
    weight_address : out std_logic_vector(12 downto 0);
    weight_read : out std_logic;
    weight_readdata : in std_logic_vector(63 downto 0);
    weight_waitrequest, weight_readdatavalid : in std_logic;
    bias_address : out std_logic_vector(6 downto 0);
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

architecture rtl of conv2_finale is
  type state_t is (IDLE,W_REQ,W_WAIT,B_REQ,B_WAIT,ROW_PREP,ROW_REQ,ROW_WAIT,
                   TAP_ADDR,TAP_READ,TAP_MAC,POOL,QUANT,WRITE_WORD,ADVANCE,FINISHED);
  signal state : state_t := IDLE;
  type rows_t is array(0 to 1007) of std_logic_vector(31 downto 0);
  signal rows : rows_t;
  signal rd_addr : integer range 0 to 1007 := 0;
  signal rd_data : std_logic_vector(31 downto 0) := (others=>'0');
  signal wr_en : std_logic := '0';
  signal wr_addr : integer range 0 to 1023 := 0;
  signal wr_data : std_logic_vector(31 downto 0) := (others=>'0');
  type weights_t is array(0 to 4607) of signed(7 downto 0);
  signal weights : weights_t;
  type biases_t is array(0 to 31) of signed(31 downto 0);
  signal biases : biases_t;
  type lanes_t is array(0 to 7) of signed(31 downto 0);
  signal acc, pooled : lanes_t := (others=>(others=>'0'));
  type quant_t is array(0 to 7) of std_logic_vector(7 downto 0);
  signal quant_bytes : quant_t := (others=>(others=>'0'));
  type m_array_t is array(0 to 31) of signed(31 downto 0);
  type s_array_t is array(0 to 31) of integer range 0 to 31;
  -- Conv2 per-channel quantization parameters (from the TFLite model).
  constant MULT : m_array_t := (
    0 => to_signed(1415663672,32),
    1 => to_signed(1756851527,32),
    2 => to_signed(1342363459,32),
    3 => to_signed(1718904837,32),
    4 => to_signed(2059456890,32),
    5 => to_signed(1675074690,32),
    6 => to_signed(1206612987,32),
    7 => to_signed(1835891667,32),
    8 => to_signed(1906908686,32),
    9 => to_signed(1132998485,32),
    10 => to_signed(1551032188,32),
    11 => to_signed(1466857349,32),
    12 => to_signed(1520389994,32),
    13 => to_signed(1870799618,32),
    14 => to_signed(1744663093,32),
    15 => to_signed(2041247547,32),
    16 => to_signed(2026039408,32),
    17 => to_signed(1235317201,32),
    18 => to_signed(1842743630,32),
    19 => to_signed(1769097232,32),
    20 => to_signed(1627325410,32),
    21 => to_signed(1725557496,32),
    22 => to_signed(1841134036,32),
    23 => to_signed(1351200012,32),
    24 => to_signed(1699059116,32),
    25 => to_signed(1684757451,32),
    26 => to_signed(1616449463,32),
    27 => to_signed(1577291085,32),
    28 => to_signed(1600713721,32),
    29 => to_signed(1611285864,32),
    30 => to_signed(1532714117,32),
    31 => to_signed(1230963156,32)
  );
  constant SHIFT : s_array_t := (
    0 => 9,
    1 => 10,
    2 => 10,
    3 => 10,
    4 => 10,
    5 => 9,
    6 => 9,
    7 => 10,
    8 => 10,
    9 => 9,
    10 => 9,
    11 => 10,
    12 => 10,
    13 => 10,
    14 => 10,
    15 => 10,
    16 => 10,
    17 => 10,
    18 => 10,
    19 => 10,
    20 => 10,
    21 => 10,
    22 => 10,
    23 => 9,
    24 => 10,
    25 => 10,
    26 => 10,
    27 => 10,
    28 => 10,
    29 => 9,
    30 => 10,
    31 => 10
  );
  constant INPUT_ZERO_POINT : integer := -128; -- VERIFY against Pool1 tensor quantization.
  constant OUTPUT_ZERO_POINT : integer := -128; -- VERIFY against Conv2 tensor quantization.
  signal wi : integer range 0 to 575 := 0;
  signal bi : integer range 0 to 31 := 0;
  signal row_slot : integer range 0 to 4 := 0;
  signal row_word : integer range 0 to 251 := 0;
  signal py, px : integer range 0 to 29 := 0;
  signal grp : integer range 0 to 3 := 0;
  signal sub : integer range 0 to 3 := 0;
  signal tap_idx : integer range 0 to 143 := 0;
  signal quant_lane : integer range 0 to 7 := 0;
  signal half_word : integer range 0 to 1 := 0;
  signal tap_byte_sel : integer range 0 to 3 := 0;
  signal tap_padding : std_logic := '0';
  signal image_addr_reg : std_logic_vector(15 downto 0) := (others=>'0');
  signal out_word : std_logic_vector(31 downto 0);
  signal count_words : unsigned(31 downto 0) := (others=>'0');
  signal busy, done : std_logic := '0';

  function requantize(a : signed(31 downto 0); c : integer) return std_logic_vector is
    variable product, nudge, tmp, high, rounded : signed(63 downto 0);
    variable rem_u, threshold : unsigned(31 downto 0);
    variable sh : integer;
    variable y : integer;
  begin
    product := a * MULT(c);
    if product(63)='0' then nudge := to_signed(1073741824,64);
    else nudge := to_signed(-1073741823,64); end if;
    tmp := product + nudge;
    if tmp(63)='1' then high := -shift_right(-tmp,31);
    else high := shift_right(tmp,31); end if;
    sh := SHIFT(c);
    if sh=0 then rounded:=high;
    else
      rem_u := unsigned(high(31 downto 0)) and
               (shift_left(to_unsigned(1,32),sh) - to_unsigned(1,32));
      threshold := shift_left(to_unsigned(1,32),sh-1)-1;
      if high(63)='1' then threshold:=threshold+1; end if;
      rounded:=shift_right(high,sh);
      if rem_u > threshold then rounded:=rounded+1; end if;
    end if;
    if rounded > to_signed(127-OUTPUT_ZERO_POINT,64) then y:=127;
    elsif rounded < to_signed(-128-OUTPUT_ZERO_POINT,64) then y:=-128;
    else y:=to_integer(rounded(31 downto 0))+OUTPUT_ZERO_POINT; end if;
    -- Conv2 has fused ReLU: clamp output at quantized zero.
    if y < OUTPUT_ZERO_POINT then y:=OUTPUT_ZERO_POINT; end if;
    return std_logic_vector(to_signed(y,8));
  end function;
begin
  avs_waitrequest <= '0';
  input_address <= image_addr_reg;
  input_read <= '1' when state=ROW_REQ else '0';
  weight_address <= std_logic_vector(to_unsigned(wi*8,13));
  weight_read <= '1' when state=W_REQ else '0';
  bias_address <= std_logic_vector(to_unsigned(bi*4,7));
  bias_read <= '1' when state=B_REQ else '0';
  output_address <= std_logic_vector(to_unsigned((py*30+px)*32+grp*8+half_word*4,16));
  output_write <= '1' when state=WRITE_WORD else '0';
  output_writedata <= out_word;
  output_byteenable <= "1111";

  process(clk)
  begin
    if rising_edge(clk) then
      rd_data <= rows(rd_addr);
      if wr_en='1' then rows(wr_addr) <= wr_data; end if;
    end if;
  end process;

  process(clk)
    variable ix, iy, ky, kx, ch, byte_addr, lane : integer;
    variable pixel, weight9 : signed(8 downto 0);
    variable prod : signed(17 downto 0);
    variable v : signed(31 downto 0);
  begin
    if rising_edge(clk) then
      wr_en<='0';
      if reset='1' then
        state<=IDLE; wi<=0; bi<=0; row_slot<=0; row_word<=0;
        py<=0; px<=0; grp<=0; sub<=0; tap_idx<=0;
        quant_lane<=0; half_word<=0; busy<='0'; done<='0';
        count_words<=(others=>'0'); rd_addr<=0;
        image_addr_reg<=(others=>'0');
        acc<=(others=>(others=>'0')); pooled<=(others=>(others=>'0'));
      else
        case state is
          when IDLE =>
            if avs_write='1' and avs_address="000" and avs_writedata(0)='1' then
              busy<='1'; done<='0'; count_words<=(others=>'0');
              wi<=0; bi<=0; row_slot<=0; row_word<=0;
              py<=0; px<=0; grp<=0; sub<=0; tap_idx<=0;
              state<=W_REQ;
            end if;
          when W_REQ =>
            if weight_waitrequest='0' then state<=W_WAIT; end if;
          when W_WAIT =>
            if weight_readdatavalid='1' then
              for i in 0 to 7 loop
                weights(wi*8+i)<=signed(weight_readdata(i*8+7 downto i*8));
              end loop;
              if wi=575 then state<=B_REQ;
              else wi<=wi+1; state<=W_REQ; end if;
            end if;
          when B_REQ =>
            if bias_waitrequest='0' then state<=B_WAIT; end if;
          when B_WAIT =>
            if bias_readdatavalid='1' then
              biases(bi)<=signed(bias_readdata);
              if bi=31 then state<=ROW_PREP;
              else bi<=bi+1; state<=B_REQ; end if;
            end if;
          when ROW_PREP =>
            if row_slot=4 then
              px<=0; grp<=0; sub<=0; tap_idx<=0; state<=TAP_ADDR;
            elsif (2*py+row_slot)<0 or (2*py+row_slot)>62 then
              row_word<=0; row_slot<=row_slot+1;
            else
              image_addr_reg<=std_logic_vector(to_unsigned((2*py+row_slot)*1008+row_word*4,16));
              state<=ROW_REQ;
            end if;
          when ROW_REQ =>
            if input_waitrequest='0' then state<=ROW_WAIT; end if;
          when ROW_WAIT =>
            if input_readdatavalid='1' then
              wr_addr<=row_slot*252+row_word;
              wr_data<=input_readdata;
              wr_en<='1';
              if row_word=251 then row_word<=0; row_slot<=row_slot+1;
              else row_word<=row_word+1; end if;
              state<=ROW_PREP;
            end if;
          when TAP_ADDR =>
            ky:=tap_idx/48;
            kx:=(tap_idx mod 48)/16;
            ch:=tap_idx mod 16;
            iy:=2*py+(sub/2)+ky;
            ix:=2*px+(sub mod 2)+kx;
            if iy<0 or iy>63 or ix<0 or ix>63 then
              tap_padding<='1'; rd_addr<=0; tap_byte_sel<=0;
            else
              tap_padding<='0';
              byte_addr:=ix*16+ch;
              rd_addr<=(sub/2+ky)*252+byte_addr/4;
              tap_byte_sel<=byte_addr mod 4;
            end if;
            state<=TAP_READ;
          when TAP_READ => state<=TAP_MAC;
          when TAP_MAC =>
            if tap_padding='1' then pixel:=to_signed(0,9);
            else
              pixel:=resize(signed(rd_data(tap_byte_sel*8+7 downto tap_byte_sel*8)),9)
                     - to_signed(INPUT_ZERO_POINT,9);
            end if;
            for l in 0 to 7 loop
              lane:=grp*8+l;
              weight9:=resize(weights(lane*144+tap_idx),9);
              prod:=pixel*weight9;
              if tap_idx=0 then v:=biases(lane)+resize(prod,32);
              else v:=acc(l)+resize(prod,32); end if;
              acc(l)<=v;
            end loop;
            if tap_idx=143 then tap_idx<=0; state<=POOL;
            else tap_idx<=tap_idx+1; state<=TAP_ADDR; end if;
          when POOL =>
            for l in 0 to 7 loop
              if sub=0 or acc(l)>pooled(l) then pooled(l)<=acc(l); end if;
            end loop;
            if sub=3 then quant_lane<=0; state<=QUANT;
            else sub<=sub+1; state<=TAP_ADDR; end if;
          when QUANT =>
            quant_bytes(quant_lane)<=requantize(pooled(quant_lane),grp*8+quant_lane);
            if quant_lane=7 then half_word<=0; state<=WRITE_WORD;
            else quant_lane<=quant_lane+1; end if;
          when WRITE_WORD =>
            if output_waitrequest='0' then
              count_words<=count_words+1;
              if half_word=0 then half_word<=1;
              else state<=ADVANCE; end if;
            end if;
          when ADVANCE =>
            sub<=0; tap_idx<=0;
            if grp=3 then
              grp<=0;
              if px=29 then
                px<=0;
                if py=29 then busy<='0'; done<='1'; state<=FINISHED;
                else py<=py+1; row_slot<=0; row_word<=0; state<=ROW_PREP; end if;
              else px<=px+1; state<=TAP_ADDR; end if;
            else grp<=grp+1; state<=TAP_ADDR; end if;
          when FINISHED =>
            if avs_write='1' and avs_address="000" and avs_writedata(0)='1' then
              state<=IDLE;
            end if;
        end case;
      end if;
    end if;
  end process;

  process(quant_bytes,half_word)
    variable p : std_logic_vector(31 downto 0);
  begin
    p:=(others=>'0');
    for i in 0 to 3 loop
      p(i*8+7 downto i*8):=quant_bytes(half_word*4+i);
    end loop;
    out_word<=p;
  end process;

  process(avs_address,avs_read,busy,done,count_words,py,px,grp)
  begin
    avs_readdata<=(others=>'0');
    if avs_read='1' then
      case avs_address is
        when "001" => avs_readdata(0)<=busy; avs_readdata(1)<=done;
        when "010" => avs_readdata<=std_logic_vector(count_words);
        when "011" => avs_readdata<=std_logic_vector(to_unsigned(py*30+px,32));
        when "100" => avs_readdata<=std_logic_vector(to_unsigned(grp*8,32));
        when others => null;
      end case;
    end if;
  end process;
end architecture;
