# Regular Expression

## Introduction of regular expression

- Describe a pattern of string matching
- Used for check whether a string contains some substring, and replace the matching string
- By using regular expression, we can:
  - Test for patterns within strings
    - For example, can test for input string to check whether it contains phone number or credit card number
  - Replace context
    - Can indicate specific context in a file using regular expression
    - And completely delete it or replace it by other context
  - Extract substring from string based on pattern matching
    - Can search for some specific context within files or input area



## Grammar of regular expression

- ### Normal characters

  - [ABC] : Match all characters in [...]

    ```javascript
    var str = "google runoob taobao";
    var patt1 = /[aeiou]/g;
    document.write(str.match(patt1));
    
    // output
    o,o,e,u,o,o,a,o,a,o
    ```

  - [^ABC] : Match all characters not in [...]

    ```js
    var str = "google runoob taobao";
    var patt1 = /[^aeiou]/g;
    document.write(str.match(patt1));
    
    // output
    g,g,l, ,r,n,b, ,t,b
    ```

  - [A-Z] : An interval of all character between them

    ```js
    var str = "Google Runoob Taobao";
    var patt1 = /[A-Z]/g;
    document.write(str.match(patt1));
    document.write("<br/");
    var patt2 = /[a-z]/g;
    document.write(str.match(patt2));
    
    // output
    G,R,T
    o,o,g,l,e,u,n,o,o,b,a,o,b,a,o
    ```

  - . : Match any single character except for line changing character (\n, \r)

    ```js
    var str = "google runoob taobao";
    var patt1 = /./g;
    document.write(str.match(patt1));
    
    // output
    g,o,o,g,l,e, ,r,u,n,o,o,b, ,t,a,o,b,a,o
    ```

  - [\s\S] : Match all. \s is matching all blank character, including line changing, \S is matching all non-blank character, not including line changing

    ```js
    var str = "google runoob taobao\nRUnoob\ntaobao";
    var patt1 = /[\s\S]/g;
    document.write(str.match(patt1));
    
    // output
    g,o,o,g,l,e, ,r,u,n,o,o,b, ,t,a,o,b,a,o, ,R,U,n,o,o,b, ,t,a,o,b,a,o
    ```

  - \w: Match all letter, number, and underline. Same as [A-Za-z0-9_]

    ```js
    var str = "Google Runoob 123Taobao";
    var patt1 = /\w/g;
    document.write(str.match(patt1));
    
    // output
    G,o,o,g,l,e,R,u,n,o,o,b,1,2,3,T,a,o,b,a,o
    ```

- ### Non-printing characters

  - \cx: Match the control character specified by x
    - For example, \cM match a Control-M or Carriage Return
    - x's value must be one of A-Z or a-z
  - \f: Match a Form feed
    - Same as \x0c and \cL
  - \n: Match a Line feed
    - Same as \x0a and \cJ
  - \r: Match an Carriage Return
    - Same as \x0d and \cM
  - \s: Match any blank character, including Space, Tabs, Form feed and so on
    - Same as [\f\n\r\t\v]
  - \S: Match any non-blank character
    - Same as [ ^ \f\n\r\t\v]
  - \t: Match any Tab
    - Same as \x09 and \cI
  - \v: Match any Vertical tab
    - Same as \x0b and \cK

- ### Special characters

  - $: Match the ending position of the input string
  - (): Label the starting and ending position of a sub-expression
  - *: Match the sub-expression before it 0 or many times
  - +: Match the sub-expression before it 1 or many times
  - .: Match any single character other than Line feed \n
  - [: Label the start of a bracket expression
  - ?: Match the sub-expression before it 0 or 1 time
  - \: Label next character as special characters, or literal characters, or backreferences, or octal escapes
  - ^: Match the starting position of the input string
  - {: Label the start of qualifier expression
  - |: Indicate the choice between two options

- ### Qualifier

  - *: Match the sub-expression before it 0 or many times

  - +: Match the sub-expression before it 1 or many times

  - ?: Match the sub-expression before it 0 or 1 time

  - {n}: Match n times, where n is a non-negative integer

  - {n,}: Match at least n times, where n is a non-negative integer

  - {n,m}: Match at least n times and at most m times, where m,n are both non-negative integers and n <= m

  - ```js
    // Match a positive integer
    // [1-9] means the first number is not 0
    // [0-9]* means have any number of numbers
    /[1-9][0-9]*/
    ```

  - ```js
    // Match any two digits positive integer
    /[0-9]{1,2}/
    ```

  - ```js
    // Match any integer between 1 ~ 99
    /[1-9][0-9]?/
    // or
    /[1-9][0-9]{0,1}/
    ```

  - ```js
    // Match all content between < and last >
    /<.*>/
    ```

  - ```js
    // Match all content between < and first >
    /<.*?>/
    // or
    /<\w+?>/
    ```

    

- ### Locator

  - ^: Match the starting position of the input string

  - $: Match the ending position of the input string

  - \b: Match a word-boundary

  - \B: Match a non-word-boundary

  - ```js
    // Match a title of a chapter followed by two digits of numbers, and occur at the beginning of the row
    /^Chapter [1-9][0-9]{0,1}/
    ```

  - ```js
    // Match a title of a chapter followed by two digits of numbers, and occur both at the beginning and end of the row
    /^Chapter [1-9][0-9]{0,1}$/
    ```

  - ```js
    // Match the substring Cha of Chapter
    /\bCha/
    ```

  - ```js
    // Match the substring ter of Chapter
    /ter\b/
    ```

  - ```js
    // Match the substring apt of Chapter but not matching words like aptitude
    /\Bapt/
    ```

- ### Select

  - Bracket all selections using (), separate adjacent selections by |

    ```js
    var str = "123456runoob123runoob456";
    var n = str.match(/([1-9])([a-z]+)/g);
    
    // output
    n: (2) ["6runoob", "3runoob"]
    ```

  - ```js
    // Search for exp1 before exp2
    /exp1(?=exp2)/
    ```

  - ```js
    // Search for exp1 after exp2
    /(?<=exp2)exp1/
    ```

  - ```js
    // Search for exp1 not followed by exp2
    /exp1(?!exp2)/
    ```

  - ```js
    // Search for exp1 not following exp2
    /(?<!exp2)exp1/
    ```



## Flags of regular expression

- ### g flag

  - Short for: global

  - Search for all matching items

  - ```js
    var str = "Google runoob taobao runoob";
    var n1 = str.match(/runoob/);
    var n2 = str.match(/runoob/g);
    
    // output
    runoob
    runoob,runoob
    ```

- ### i flag

  - Short for: ignore

  - Ignore the difference between upper and lower case when searching

  - ```js
    var str = "Google runoob taobao RUNoob";
    var n1 = str.match(/runoob/g);
    var n2 = str.match(/runoob/gi);
    
    // output
    runoob
    runoob,RUNoob
    ```

- ### m flag

  - Short for: multi line

  - Let ^ and $ match every line's beginning and end in a context

  - ```js
    var str = "runoobgoogle\ntaobao\nrunoobweibo";
    var n1 = str.match(/^runoob/g);
    var n2 = str.match(/^runoob/gm);
    
    // output
    runoob
    runoob,runoob
    ```

- ### s flag

  - Short for: special character . contains \n

  - Generally, . is to match any character other than \n, by adding s flag, . contains \n

  - ```js
    var str = "google\nrunoob\ntaobao";
    var n1 = str.match(/google./);
    var n2 = str.match(/runoob./s);
    
    // output
    null
    runoob
    ```



## Metacharacter

- \: Label next character as special characters, or literal characters, or backreferences, or octal escapes

- ^: Match the starting position of the input string

- $: Match the ending position of the input string

- *: Match the sub-expression before it 0 or many times

- +: Match the sub-expression before it 1 or many times

- ?: Match the sub-expression before it 0 or 1 time

- {n}: Match n times, where n is a non-negative integer

- {n,}: Match at least n times, where n is a non-negative integer

- {n,m}: Match at least n times and at most m times, where m,n are both non-negative integers and n <= m

- ?: When this character follows after any other limiter (*, +, ?, {n}, {n,}, {n,m}), the matching mode is ungreedy

- .: Match any single character other than Line feed \n

- (pattern): Match pattern and get such matching

- (?:pattern): Match pattern but does not get result

- (?=pattern): Look ahead positive assert, starting to search for string at any string that matches pattern

- (?!pattern): Look ahead negative assert, starting to search for string at any string that does not match pattern

- (?<=pattern): Look behind positive assert, similar to look ahead positive assert but with opposite direction

- (?<!pattern): Look behind negative assert, similar to look ahead negative assert but with opposite direction

- x|y: Match x or y

- [xyz] : Match any character it contains

- [^xyz] : Match any character it does not contain

- [a-z] : Match any character in this interval

- [^a-z] : Match any character not in this interval

- \b: Match a word-boundary

- \B: Match a non-word-boundary

- \cx: Match the control character specified by x

- \d: Match a digit, same as [0-9]

- \D: Match a non-digit, same as [ ^ 0-9]

- \f: Match a Form feed, same as \x0c and \cL

- \n: Match a Line feed, same as \x0a and \cJ

- \r: Match an Carriage Return, same as \x0d and \cM

- \s: Match any blank character, including Space, Tabs, Form feed and so on, same as [\f\n\r\t\v]

- \S: Match any non-blank character, same as [ ^ \f\n\r\t\v]

- \t: Match any Tab, same as \x09 and \cI

- \v: Match any Vertical tab, same as \x0b and \cK

- \w: Match any letter, digit, underline, same as [A-Za-z0-9_]

- \W: Match any character that are not letter, digit, underline, same as [ ^ A-Za-z0-9_]

- \xn: Match n where n is a hex escape value

- \num: Match num where num is a positive integer

- \n: Identify an octal escape value or a backreference

- \nm: Identify an octal escape value or a backreference

- \nml: If n is a octal value (0-3), and both m and l are octal values (0-7), match octal escape value nml

- \un: Match n where n is a Unicode character that represented by four hex numbers

- ```js
  var str = "abcd test@runoob.com 1234";
  var patt1 = /\b[\w.%+-]+@[\w.-]+\.[a-zA-Z]{2,6}\b/g;
  document.write(str.match(patt1));
  
  // output
  test@runoob.com
  ```



## Operator priority

- Priority from highest to lowest:
  - \
  - (), (?:), (?=), []
  - *, +, ?, {n}, {n,}, {n,m}
  - ^, $, \any metacharacter or any character
  - |

