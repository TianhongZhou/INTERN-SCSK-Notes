# Common UNIX/Linux command

## File and directory command

- ### Browse directory command --

  1. ​	Command:	  ls

     ​	Function:	     Show directory file

     ​	Grammar:		ls parameter[-ald] [file or content]

     ​							-a:	Show all files, include hidden files

     ​							-l:	  Detailed information

     ​							-d:	 View directory properties

  2. ​    Command:      pwd

     ​    Function:         Show current working directory

     ​    Grammar:        pwd

     

- ### Directory manipulation command --

  1. ​	 Command:      cd

     ​     Function:         Change directory

     ​     Grammar:        cd [directory]

  2. ​     Command:       mkdir

     ​     Function:          Make new directory

     ​     Grammar:         mkdir [directory name]

  3. ​     Command:        rmdir

     ​     Function:           Remove empty directory

     ​     Grammar:          rmdir [directory name]

     

- ### Browse file command --

  1. ​      Command:       cat

     ​      Function:          Concatenate file and display to standard output device

     ​      Grammar:         cat parameter[-En] [file]

     ​                  			 -E:	Display $ at the end of each line

     ​							   -n:	Add line number for each line

  2. ​       Command:       more(less)

     ​       Function:          Display file contents in pagination (use up and down to roll page)

     ​       Grammar:         more(less) [file]

     ​								Space or f to display next page

     ​                                Enter to display next line

     ​                                 q or Q to quit

  3. ​        Command:        head

     ​        Function:           Display the first few lines of the file

     ​        Grammar:          head parameter[-n] [file]

     ​                                  -n:     Display first n lines, if no input then display first 10 lines

  4. ​         Command:        tail

     ​         Function:           Display the last few lines of the file

     ​         Grammar:          tail parameter[-nF] [file]

     ​                                   -n:     Display last n lines, if no input then display last 10 lines

     ​                                   +n:     Display from the n th line to the end of the file

     ​                                   -F:     Use to trace display end-growing file

     

- ### File manipulation command --

  1. ​          Command:       cp

     ​          Function:          Copy a file from one place to another

     ​          Grammar:         cp parameter[-piru] [origin directory] [target directory]

     ​                                   -p:     Copy along with the file's attributes instead of using default ways, often used for backups

     ​								   -i:      If target file already exists, when overwriting, the operation will be asked first

     ​                                   -r:      Recursive persistent replication for directory replication behavior

     ​                                   -u:     Only copy when origin and target file are different

  2. ​          Command:        rm

     ​          Function:           Remove file or directory

     ​          Grammar:          rm parameter[-fir] [target file path]

     ​                                    -f:     Meaning of force, ignore non-existing files without warning message

     ​                                    -i:     Interactive mode, the user will be asked whether to operate before deleting

     ​                                    -r:     Recursive deletion, commonly used for delete directory

  3. ​           Command:        find

     ​           Function:           Find file or directory

     ​           Grammar:          find [path] parameter[-name/size/type/perm mode]

     ​                                     -name [filename]:     Find the file with the name

     ​                                     -size [+-]SIZE:          Find file with size larger/smaller than SIZE

     ​                                     -type TYPE:              Find file with the type of TYPE: 

     ​                                                                       General file (f)

     ​                                                                       Device file (b, c)

     ​                                                                       Directory (d)

     ​                                                                       Link file (l)

     ​                                                                       socket (s)

     ​                                                                       FIFO pipe file (p)

     ​                                       -perm mode:            Find file with permissions just equal to mode which represented by numbers 

  4. ​            Command:         grep

     ​            Function:            Search for matched string in the file and print

     ​            Grammar:           grep parameter[-aciv] [target string] [file to find string]

     ​                                       -a:     Find data in binary file as text file

     ​                                       -c:     Count the number of times the target string was found

     ​                                       -i:      Ignore the difference in case of letters

     ​                                       -v:     Inverse selection, that is, display the line that does not contain the target string

  5. ​           Command:          tar

     ​           Function:             Pack a file or directory specified by the user into a file, or open the compression/decompression function by specifying parameters

     ​           Grammar:            tar parameter[-ctxjzvfC] [file]

     ​                                       -c:                Create new package file

     ​                                       -t:                 Check what file names does the package file contain

     ​                                       -x:                Decompress, can specify the directory for decompressing using -C together

     ​                                       -j:                 Use bzip2 to support compress or decompress

     ​                                       -z:                Use gzip to support compress or decompress  

     ​                                       -v:                Show the name of current processing file during compression or decompression

     ​                                       -f [name]:     Create archive with given filename 

     ​                                       -C dir:           Specify the directory for compressing or decompressing

     

## Process and control command

- ### View system processes command --

  1. ​           Command:          ps

     ​           Function:             Display the instantaneous process information of the system, he can display the system process and process related information when the user enters this command

     ​           Grammar:            ps parameter[-lujfaxr]

     ​                                       -l:      Long format output

     ​                                       -u:     Display process in order of username and startup time

     ​                                       -j:      Display process in task format

     ​                                       -f:      Display process in tree format

     ​                                       -a:     Display all process of all users

     ​                                       -x:     Display process without terminal control

     ​                                       -r:      Display process in progress

  2. ​           Command:          top

     ​           Function:             Tool for dynamic monitoring of system tasks, output is continuous

     ​           Grammar:            top parameter[-bcdinpqsS]

     ​                                       -b:              Run in batch mode, but cannot accept command line input

     ​                                       -c:              Display command line instead of only command name

     ​                                       -d N:          Display the delay time between screen updates

     ​                                       -i:               Suppress idle or zombie processes

     ​                                       -n NUM:     Display NUM times of updated data, then quit

     ​                                       -p PID:       Monitor only the id of the specified process

     ​                                       -q:              Refresh without any delay

     ​                                       -s:              Run in safe mode, disable some interoperability commands

     ​                                       -S:              Cumulative mode, output the total CPU time of each process



- ### Control system processes command --

  1. ​           Command:          kill

     ​           Function:             Send a signal to a process (identified by PID), usually used with ps and jobs commands

     ​           Grammar:            kill parameter[-signal] PID

     ​                                       -SIGHUP:        Start the terminated process

     ​                                       -SIGINT:          Same as ctrl+c, interrupt the progress of a program

     ​                                       -SIGKILL:        Forcibly interrupt the progress of a process

     ​                                       -SIGTERM:     Terminate the process in the normal way of ending the process

     ​                                       -SIGSTOP:      Same as ctrl+z, stop the 

  2. ​           Command:          killall

     ​           Function:             Use the name of process to kill it, this command can kill a group of processes with the same name

     ​           Grammar:            killall parameter[-eIpilqrsu] [name of running process]

     ​                                       -e:     Exact match for long name

     ​                                       -I:      Ignore the difference of cases of letters

     ​                                       -p:     Kill the process group to which the process belongs

     ​                                       -i:      Kill the process interactively, need to confirm before killing the process

     ​                                       -l:      Print a list of all known signals

     ​                                       -q:     If there is no process being killed, then return nothing

     ​                                       -r:      Use regular expressions to match process name to kill

     ​                                       -s:     Replace the default signal SIGTERM with the specified PID

     ​                                       -u:     Kill the processes of a specified user

  3. ​           Command:          nice

     ​           Function:             Increase or decrease the priority from default priority and run the command with new priority

     ​           Grammar:            nice parameter[-n --adjustment] [filename]

     ​                                       -n --adjustment:     Specify the adjustment value of the process running priority, from -20 to 19

  4. ​           Command:          renice

     ​           Function:             Change the nice value of a running process

     ​           Grammar:            renice parameter[-n] [PID]

     ​                                       -n:     Specify the adjustment value of the process running priority

     

- ### Run program in background command --

  1. ​           Command:          &

     ​           Function:             Run program in background



- ### Suspending and resuming process command --

  1. ​           Command:          ctrl+z

     ​           Function:             Suspend process

  2. ​           Command:          ctrl+c

     ​           Function:             End process

  3. ​           Command:          fg[n]

     ​           Function:             Revert to the foreground and continue running

  4. ​           Command:          bg[n]

     ​           Function:             Revert to the background and continue running

  5. ​           Command:          jobs

     ​           Function:             Check suspended process



## User and  permission management command

- ### User management command --

  1. ​           Command:          useradd

     ​           Function:             Add new user account (superuser available)

     ​           Grammar:           useradd parameter[-degGsu] [username]

     ​                                       -d:     Specify the user's home directory when logging in

     ​                                       -e:     Account termination date

     ​                                       -g:     Specify the user group to which the account belongs

     ​                                       -G:    Specify the additional group to which the account belongs

     ​                                       -s:     Specify the shell used by the account after logging in

     ​                                       -u:     Specify the user's ID
  
  2. ​           Command:          passwd
  
     ​           Function:             Set or change user's password and password attributes
  
     ​           Grammar:            passwd parameter[-dlus] [username]
  
     ​                                       -d:     Delete user's password
  
     ​                                       -l:      Temporarily lock the specified user account
  
     ​                                       -u:     Unlock the specified user account
  
     ​                                       -s:     Show the status of specified user account
  
  3. ​           Command:          usermod
  
     ​           Function:             Modify user properties (superuser available)
  
     ​           Grammar:            usermod parameter[-degGsul] [username]
  
     ​                                       -d:     Specify the user's home directory when logging in
  
     ​                                       -e:     Account termination date
  
     ​                                       -g:     Specify the user group to which the account belongs
  
     ​                                       -G:    Specify the additional group to which the account belongs
  
     ​                                       -s:     Specify the shell used by the account after logging in
  
     ​                                       -u:     Specify the user's ID
  
     ​                                       -l:      Change to new username
  
  4. ​           Command:          userdel
  
     ​           Function:             Delete specified user account (superuser available)
  
     ​           Grammar:            userdel parameter[-rf] [username]
  
     ​                                       -r:      Delete the user account, along with its home directory and local mail storage directory or file
  
     ​                                       -f:      Delete the user login directory and all files in the directory
  
  5. ​           Command:          su
  
     ​           Function:             Switch user
  
     ​           Grammar:            su [username]
  
  6. ​           Command:          id
  
     ​           Function:             Display user's UID, GID and information of user group to which the user belongs to, if no specified user, then display current user information
  
     ​           Grammar:            id [username]
  
  7. ​           Command:          whoami
  
     ​           Function:             View current username
  
  8. ​           Command:          w
  
     ​           Function:             View currently logged in system users and details



- ### User  group management command --

  1. ​           Command:          groupadd

     ​           Function:             Add new user group (superuser available)

     ​           Grammar:           groupadd parameter[-go] [username]

     ​                                      -g:     Specify user group ID

     ​                                      -o:     Allow group ID to be non-unique

  2. ​           Command:          groupmod

     ​           Function:             Modify user group properties (superuser available)

     ​           Grammar:            groupmod parameter[-gno] [username]

     ​                                       -g:     Specify new user group ID

     ​                                       -n:     Specify new user group name

     ​                                       -o:     Allow group ID to be non-unique

  3. ​           Command:          groupdel

     ​           Function:             Delete specified user group (superuser available)

     ​           Grammar:            groupdel [username]



- ### File permission management command --

  1. ​           Command:          chmod

     ​           Function:             Modify file access permission

     ​           Grammar:            chmod parameter[object, operator, permission] [file]

     ​                                       Object:          u:     File owner

     ​                                                            g:      Users in the same group

     ​                                                            o:      Users in other groups

     ​                                       Operator:      +:      Add

     ​                                                             -:      Delete

     ​                                                            =:      Give

     ​                                       Permission:   r:      Read

     ​                                                             w:     Write

     ​                                                             x:      Execution

     ​                                                             s:      Set user ID

  2. ​           Command:          chown

     ​           Function:             Change the owner of specified file to specified user or user group

     ​           Grammar:            chown parameter[-cfhRv/deference] [owner/owner group] [file]

     ​                                       -c:                   Display information of changed part

     ​                                       -f:                    Ignore incorrect information

     ​                                       -h:                   Fix symlinks

     ​                                       -R:                  Process all files in the specified directory and its subdirectories

     ​                                       -v:                   Display detailed process information

     ​                                       -deference:     Acts on the point of a symlink, not the symlink itself

  3. ​           Command:          chgrp

     ​           Function:             Change the owner user group to which file belongs

     ​           Grammar:            chgrp parameter[-cfRv/deference/no-deference] [user group] [file]

     ​                                       -c:                       Output debug information when changes are made

     ​                                       -f:                        Not display incorrect information

     ​                                       -R:                       Process all files in the specified directory and its subdirectories

     ​                                       -v:                       Display detailed process information when running

     ​                                       -deference:         Acts on the point of a symlink, not the symlink itself

     ​                                       -no-deference:    Acts on the symlink itself