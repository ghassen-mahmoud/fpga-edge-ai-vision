library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Eight parallel INT8 products accumulated into one INT32 sum.
-- Input sample is centered signed 9-bit (input_int8 - input_zero_point).
-- Each accepted valid cycle adds eight products; clear starts a new dot product.
-- acc_out is registered and valid one clock after the corresponding input.
entity mac8_int8 is
  port (
    clk      : in std_logic;
    reset    : in std_logic;
    clear    : in std_logic;
    valid_in : in std_logic;
    x_packed : in std_logic_vector(71 downto 0);
    w_packed : in std_logic_vector(63 downto 0);
    acc_out  : out std_logic_vector(31 downto 0);
    valid_out: out std_logic
  );
end entity;

architecture rtl of mac8_int8 is
  signal acc : signed(31 downto 0) := (others=>'0');
  signal vreg : std_logic := '0';
begin
  acc_out <= std_logic_vector(acc);
  valid_out <= vreg;
  process(clk)
    variable total : signed(31 downto 0);
    variable x : signed(8 downto 0);
    variable w : signed(7 downto 0);
    variable prod : signed(16 downto 0);
  begin
    if rising_edge(clk) then
      if reset='1' then
        acc <= (others=>'0');
        vreg <= '0';
      else
        vreg <= valid_in;
        if clear='1' then
          total := (others=>'0');
        else
          total := acc;
        end if;
        if valid_in='1' then
          for i in 0 to 7 loop
            x := signed(x_packed(i*9+8 downto i*9));
            w := signed(w_packed(i*8+7 downto i*8));
            prod := x * w;
            total := total + resize(prod,32);
          end loop;
        end if;
        acc <= total;
      end if;
    end if;
  end process;
end architecture;
