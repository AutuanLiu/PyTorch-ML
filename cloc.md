# CLOC helper

Invoke cloc to count your source files, directories, archives, or git commits.

- [CLOC helper](#cloc-helper)
    - [Examples](#examples)

## Examples
1. a file
    ```bash
    $ cloc hello.c
    ```
2. a directory
    ```bash
    $ cloc gcc-5.2.0/gcc/c
    ```
3. an archive
    ```bash
    $ cloc master.zip
    ```
4. a git repository, using a specific commit
    ```bash
    $ cloc 6be804e07a5db
    ```
5. each subdirectory of a particular directory
    ```bash
    $ for d in ./*/ ; do (cd "$d" && echo "$d" && cloc --vcs git); done
    ```
6. diff
    ```bash
    $ cloc --diff Python-2.7.9.tgz Python-2.7.10.tar.xz
    ```