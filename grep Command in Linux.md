# grep Command in Linux

## What is grep command?

- Short for: global regular expressions print
- Search one or more files for a specific character pattern (that is, a regular expression), which can be a single character, string, word or sentence

## How to use grep command?

- Grammar: grep [-abcEFGhHilLnqrsvVwxy] [-A<Display row numebr>] [-B<Display row number>] [-C<Display row number>] [-d<Action>] [-e<Template style>] [-f<Template file>] [--help] [Template style] [file or directory]
- Parameters:
  - -a or --text: Do not ignore data in binary
  - -A<Display row number> or --after-context=<Display row number>: In addition to showing the row that conforms to the template style, show the content after that row
  - -b or --byte-offset: Before showing the row that conforms to the template style, label the number of the first character of that row
  - -B<Display row number> or --before-context=<Display row number>: In addition to showing the row that conforms to the style, show the content before that row
  - -c or --count: Count the number of columns that conforms to the style
  - -C<Display row number> or - context=<Display row number>: In addition to showing the row that conforms to the style, show the content both before and after that row
  - -d<Action> or --directions=<Action>: If what we want to search for is directory instead of file, we must use this parameter
  - -e<Template style> or --regexp=<Template style>: Indicate string as the style of searching file context
  - -E or --extended-regexp: Use the style as an extended regular expression
  - -f<Rule file> or --file=<Rule file>: Indicate rule file which contains one or more rule style. Let grep search for the file context which conforms the rules and conditions
  - -F or --fixed-regexp: Treat styles as a list of fixed strings
  - -G or --basic-regexp: Treat styles as a normal expression
  - -h or --no-filename: Do not indicate the name of the file to which the line belongs before the line that matches the style is displayed
  - -H or --with-filename: Indicate the name of the file to which the line belongs before the line that matches the style is displayed
  - -i or --ignore-case: Ignore the difference between upper and lower cases
  - -l or --file-with-matches: List the file names which file context conforms the specific style
  - -L or  --file-without-matches: List the file names which file context does not conform the specific style
  - -n or  --line-number: Label the row number of this line before the line that matches the style is displayed
  - -o or --only-matching: Only show the part that matched PATTERN
  - -q or --quiet and --silent: Do not display any information
  - -r or --recursive: Same as using parameter -d recurse
  - -s or --no-message: Do not display incorrect information
  - -v or  --invert-match: Display all rows which do not contain matching context
  - -V or --version: Display the version information
  - -w or --word-regexp: Only display the row which matches the whole word
  - -x or --line-regexp: Only display the row which matches the whole row
  - -y Same as parameter -i