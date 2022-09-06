# sed Command in Linux

## What is sed command?

- Short for: stream editor
- Row-Oriented tool
- Process every row and return the output to STDOUT

## How does sed work?

- Process the data in text file based on scripts
- These scripts are either inputted in command or stored in a file
- Sequence of processing data:
  - Only read one row each time
  - Change data according to provided rules
  - Will not change the file directly
  - Will copy the file and only change the copied one
  - Return output

## How to use sed command?

- Grammar: sed [-hnV] [-e<script>] [-f<script file>] [text file]
- Parameters:
  - -e<script> or --expression=<script>: Use provided specific script to process input text file
  - -f<script file> or --file=<script file>: Use provided specific scripts file to process input text file
  - -h or --help: Show help
  - -n or -quiet and --silent: Only show the result after processed by script
  - -V or --version: Show version
  - -i: Change content of file directly
- Actions:
  - a:
    - Add
    - Can be followed by string
    - Add these strings to a new row after current
  - c:
    - Replace
    - Can be followed by string
    - Replace these strings to the row between n1 and n2
  - d:
    - Delete
    - Usually be followed by nothing
  - i:
    - Insert
    - Can be followed by string
    - Add these strings to a new row before current
  - p:
    - print
    - Print some data out

## Examples of using sed command

- Origin file:

  ```
  $ cat testfile
  --------------------------------------------
  HELLO LINUX!
  Linux is a free unix-type opterating system.
  This is a linux testfile!
  Linux test
  Google
  Taobao
  Runoob
  Testfile
  Wiki
  ```

- Add a line after the forth line and return the output to STDOUT

  ```
  $ sed -e 4a\newLine testfile
  --------------------------------------------
  HELLO LINUX!
  Linux is a free unix-type opterating system.
  This is a linux testfile!
  Linux test
  newLine
  Google
  Taobao
  Runoob
  Testfile
  Wiki
  ```

- Show testfile content and the line number, also delete row 2 to 5

  ```
  $ nl testfile | sed '2,5d'
  --------------------------------------------
  	1	HELLO LINUX!
  	6	Taobao
  	7	Runoob
  	8	Testfile
  	9	Wiki
  ```

- Delete row 3 to end

  ```
  $ nl testfile | sed '3,$d'
  --------------------------------------------
  	1	HELLO LINUX!
  	2	Linux is a free unix-type opterating system.
  ```

- Add drink tea after the second line

  ```
  $ nl testfile | sed '2a drink tea'
  --------------------------------------------
  	1	HELLO LINUX!
  	2	Linux is a free unix-type opterating system.
  drink tea
  	3	This is a linux testfile!
  	4	Linux test
  	5	Google
  	6	Taobao
  	7	Runoob
  	8	Testfile
  	9	Wiki
  ```

- Add drink tea before the second line

  ```
  $ nl testfile | sed '2i drink tea'
  --------------------------------------------
  	1	HELLO LINUX!
  drink tea
  	2	Linux is a free unix-type opterating system.
  	3	This is a linux testfile!
  	4	Linux test
  	5	Google
  	6	Taobao
  	7	Runoob
  	8	Testfile
  	9	Wiki
  ```

- Add two lines after the second line

  ```
  $ nl testfile | sed '2a Drink tea or ...\
  drink beer'
  --------------------------------------------
  	1	HELLO LINUX!
  	2 	Linux is a free unix-type opterating system.
  Drink tea or ...
  drink beer?
  	3	This is a linux testfile!
  	4	Linux test
  	5	Google
  	6	Taobao
  	7	Runoob
  	8	Testfile
  	9	Wiki
  ```

- Replace line 2 to 5 with No 2-5 number

  ```
  $ nl testfile | sed '2,5c No 2-5 number'
  --------------------------------------------
  	1	HELLO LINUX!
  No 2-5 number
  	6	Taobao
  	7	Runoob
  	8	Testfile
  	9	Wiki
  ```

- Search for the rows with oo in testfile

  ```
  $ nl testfile | sed -n '/oo/p'
  --------------------------------------------
  	5	Google
  	7	Runoob
  ```

- Search for the rows with oo and do the scripts

  ```
  $ nl testfile | sed -n '/oo/{s/oo/kk/;p;q}'
  --------------------------------------------
  	5	Gkkgle
  ```

- Replace oo by kk at the first time it occurs in each row

  ```
  sed -e 's/oo/kk/' testfile
  ```

- Replace all oo by kk in the whole file

  ```
  sed -e 's/oo/kk/g' testfile
  ```

- Using one sed do two things: delete the third row to the end, and replace HELLO by RUNOOB

  ```
  $ nl testfile | sed -e '3,$d' -e 's/HELLO/RUNOOB'
  --------------------------------------------
  	1	RUNOOB LINUX!
  	2	Linux is a free unix-type opterating system.
  ```

- sed can also directly change the file instead of copying it first

- Origin file:

  ```
  $ cat regular_express.txt
  --------------------------------------------
  runoob.
  google.
  taobao.
  facebook.
  zhihu-
  weibo-
  ```

- If any row ended by . , replace it to !

  ```
  $ sed -i 's/\.$/\!/g' regular_express.txt
  $ cat regular_express.txt
  --------------------------------------------
  runoob!
  google!
  taobao!
  facebook!
  zhihu-
  weibo-
  ```

  