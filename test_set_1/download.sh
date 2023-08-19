#!/bin/bash
BASEDIR=$(dirname "$0")

printf "Fetching Test Set 1"
printf "\nFetching test_set_1.zip$i"
git restore --source origin/test_set_1 $BASEDIR/test_set_1.zip
for (( i=1; i<10; i++ ))
do
    printf "\nFetching test_set_1.z0$i"
    git restore --source origin/test_set_1 $BASEDIR/test_set_1.z0$i
done

printf "\nVerify data using\n"
printf "=================\n"
echo "cd $BASEDIR && md5sum -c checksum && cd `pwd`"

printf "\nUse your OS zip extractor OR the following command:\n"
printf "===================================================\n"
printf "zip -s0 $BASEDIR/test_set_1.zip --out temp.zip && unzip temp.zip -d $BASEDIR/../data/MicrosoftDNS_4_ICASSP/ && rm temp.zip\n"