# awk Command in Linux

## What is awk command?

- A language to process context file
- A tool of context analysis

## How to use awk command?

- Grammar: awk [parameter] 'script' var=value file

  ​			or: awk [parameter] -f scriptfile var=value file

- Parameters:
  - -F fs or --field-separator fs: Specify the input file fold delimiter, fs is a string or a regular expression
  - -v var=value or --asign var=value: Assign a user-defined variable
  - -f scriptfile or  --file scriptfile: Read awk command from script file
  - -mf nnn or -mr nnn: Sets an intrinsic limit on the nnn value
    - -mf option limits the maximum number of blocks allocated to nnn
    - -mr option limits the maximum number of records
  - -W compact or --compat, -W traditional or --traditional: Run awk in compatibility mode
  - -W copyleft or --copyleft, -W copyright or --copyright: Print a short copyright message
  - -W help or --help, -W usage or --usage: Print all awk option and a short explanation of each option
  - -W lint or --lint: Print warning for constructs that are not portable to legacy unix platforms
  - -W lint-old or --lint-old: Print warning about constructs that are not portable to legacy unix platforms
  - -W posix: Open compatibility mode
  - -W re-interval or --re-interval: Allow the use of interval regular expressions
  - -W source program-text or --source program-text: Use program-text as source code
  - -W version or --version: Print version of bug report information

- Basic examples:

  - Context of log.txt

    ```
    2 this is a test
    3 Do you like awk
    This's a test
    10 There are orange,apple,mongo
    ```

  - Using space or Tab to separate every row, return the first and forth items

    ```
    $ awk '{print $1,$4}' log.txt
    --------------------------------------------
    2 a
    3 like
    This's
    10 orange,apple,mongo
    
    $ awk '{print "%-8s %-10s\n",$1,$4}' log.txt
    --------------------------------------------
    2		a
    3		like
    This's
    10		orange,apple,mongo
    ```

  - Using , to separate

    ```
    $ awk -F, '{print $1,$2}' log.txt
    --------------------------------------------
    2 this is a test
    3 Do you like awk
    This's a test
    10 There are orange apple
    ```

  - Using space to separate first, and then using , to separate result

    ```
    $ awk -F '[ ,]' '{print $1,$2,$5}' log.txt
    --------------------------------------------
    2 this test
    3 Are awk
    This's a 
    10 There apple
    ```

  - Set variable

    ```
    $ awk -va=1 '{print $1,$1+a}' log.txt
    --------------------------------------------
    2 3
    3 4
    This's 1
    10 11
    
    $ awk -va=1 -vb=s '{print $1,$1+a,$1b}' log.txt
    --------------------------------------------
    2 3 2s
    3 4 3s
    This's 1 This'ss
    10 11 10s
    ```

- Operator

  - = += -= *= /= %= ^= **=: assign value

  - ?: : C condition expressions

  - ||: or

  - &&: and

  - ~ !~: Match regular expression and not match regular expression

  - < <= > >= != ==: relationship operator

  - space: connect

  - +-: add, minus

  - */%: times, divide, mod

  - ^ ***: exponential

  - ++ --: increase, decrease

  - $: field reference

  - in: array member

  - Filter rows where the first column is greater than 2

    ```
    $ awk '$1>2' log.text
    --------------------------------------------
    3 Do you like awk
    This's a test
    10 There are orange,apple,mongo
    ```

  - Filter rows where the first column is equal to 2

    ```
    $ awk '$1==2 {print $1,$3}' log.txt
    --------------------------------------------
    2 is
    ```

  - Filter rows where the first column is greater than 2 and the second column is equal to 'Are'

    ```
    $ awk '$1>2 && $2=="Are" {print $1,$2,$3}' log.txt
    --------------------------------------------
    3 Are you
    ```

- Built-in variable

  - $n: The nth field of the current record, separated by FS

  - $0: Complete input record

  - ARGC: Number of command parameters

  - ARGIND: Position of current file in command

  - ARGV: Array that contains command parameters

  - CONVFMT: Number conversion format (default is %.6g) ENVIRON environment variable associative array

  - ERRNO: The last description of system error

  - FIELDWIDTHS: List of field widths (space separated)

  - FILENAME: Current file name

  - FNR: Line number counted separately for each file

  - FS: Field separator (default is any space)

  - IGNORECASE: If it is true, ignore upper and lower cases when matching

  - NF: The number of fields in a record

  - NR: The number of records that have been read, which is the line number, starting from 1

  - OFMT: Output format for numbers (default is %.6g)

  - OFS: The output field separator, the default value is the same as the input field separator

  - ORS: The output record separator (default is a newline)

  - RLENGTH: The length of the string matched by the match function

  - RS: Record separator (default is a newline)

  - RSTART: The first position of the string matched by the match function

  - SUBSEP: Array subscript separator (default is /034)

  - Return the row with second column contains th, and print the second and forth column

    ```
    $ awk '$2 ~ /th/ {print $2,$4}' log.txt
    --------------------------------------------
    this a
    ```

  - Return the row contains re

    ```
    $ awk '/re/' log.txt
    --------------------------------------------
    10 There are orange,apple,mongo
    ```

  - Ignore upper and lower case

    ```
    $ awk 'BEGIN{IGNORECASE=1} /this/' log.txt
    --------------------------------------------
    2 this is a test
    This's a test
    ```

  - Mode negation

    ```
    $ awk '$2 !~ /th/ {print $2,$4}' log.txt
    --------------------------------------------
    Are like
    a
    There orange,apple,mongo
    
    $ awk '!/th/ {print $2,$4}' log.txt
    --------------------------------------------
    Are like
    a
    There orange,apple,mongo
    ```

- awk scripts

  - Suppose we have such a file

    ```
    $ cat score.txt
    --------------------------------------------
    Marry		2143 78 84 77
    Jack		2321 66 78 45
    Tom			2122 48 77 71
    Mike		2537 87 97 95
    Bob			2415 40 57 62
    ```

  - Scripts:

    ```
    $ cat cal.awk
    --------------------------------------------
    #!/bin/awk -f
    
    BEGIN {
    	math = 0
    	english = 0
    	computer = 0
    	
    	printf "NAME		NO.	MATH ENGLISH COMPUTER TOTAL\n"
    	printf "--------------------------------------------\n"
    }
    
    {
    	math += $3
    	englist += $4
    	computer += $5
    	printf "%-6s %-6s %4d %8d %8d %8d\n", $1, $2, $3,$4$5, $3+$4+$5
    }
    
    END {
    	printf "--------------------------------------------\n"
    	printf "  TOTAL:%10d %8d %8d \n", math, english, computer
    	printf "AVERAGE:%10.2f %8.2f %8.2f\n", math/NR, english/NR, computer/NR
    }
    ```

  - Output:

    ```
    $ awk -f cal.awk score.txt
    --------------------------------------------
    NAME		NO.	 MATH ENGLISH COMPUTER TOTAL
    --------------------------------------------
    Marry	   2143    78      84       77  239
    Jack	   2321    66      78       45  189
    Tom		   2122    48      77       71  196
    Mike	   2537    87      97       95  279
    Bob		   2415    40      57       62  159
    --------------------------------------------
      TOTAL:           319     393      350
    AVERAGE:         63.80   78.60    70.00
    ```

    