# aria2 download Helper

- [aria2 download Helper](#aria2-download-helper)
    - [website](#website)
    - [help](#help)
    - [Examples](#examples)

## website
1. [aria2](https://aria2.github.io/)

## help
1. 格式
    ```bash
    $ aria2c [OPTIONS] [URI | MAGNET | TORRENT_FILE | METALINK_FILE]
    
2. optiions
    ```bash
    -d, --dir=DIR                The directory to store the downloaded file.
    -o, --out=FILE               The file name of the downloaded file.
    -s, --split=N                Download a file using N connections.
    -i, --input-file=FILE        Downloads URIs found in FILE.
    -j                            Set maximum number of parallel downloads for every static (HTTP/FTP) URL, torren and metalink.
    
    ```
## Examples
1. Download from WEB

    ```bash
    $ aria2c http://example.org/mylinux.iso
    ```

2. Download from 2 sources

    ```bash
    $ aria2c http://a/f.iso ftp://b/f.iso
    ```
3. Download using 2 connections per host

    ```bash
    $ aria2c -x2 http://a/f.iso
    ```

4. BitTorrent

    ```bash
    $ aria2c http://example.org/mylinux.torrent
    ```

5. BitTorrent Magnet URI

    ```bash
    $ aria2c 'magnet:?xt=urn:btih:248D0A1CD08284299DE78D5C1ED359BB46717D8C'
    ```

6. Metalink

    ```bash
    $ aria2c http://example.org/mylinux.metalink
    ```

7. Download URIs found in text file:

    ```bash
    $ aria2c -i uris.txt
    ```