# README

## Test set 1 audio samples

<!-- * Created using
    `zip ../test_set_1.zip --out test_set_1.zip -s 3g` -->

* Get the test_set data using
    `./download.sh`
    > Note: The test set download makes use of git large file system (GIT LFS). Make sure you have installed git-lfs `git lfs install`
    
> The download script will output the instructions to verify the download and extract the dataset.
    
* Verify the files using
    `md5sum -c checksum`

* Extract using
    `zip -s0 test_set_1.zip --out temp.zip && unzip temp.zip -d ../data && rm temp.zip`
    Or double click `test_set_1.zip` and unzip using your OS zip extractor (7zip, winzip, etc.)
