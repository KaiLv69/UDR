#### Downloading ####
ES=./elasticsearch-7.9.1/bin/elasticsearch
if test -f "$ES"; then
    echo "$ES exists. Using the existent one"
else 
    echo "$ES does not exist. Downloading a new one"
    wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.9.1-linux-x86_64.tar.gz
    tar -xf elasticsearch-7.9.1-linux-x86_64.tar.gz
fi

#### Starting the ES service ####
nohup ./elasticsearch-7.9.1/bin/elasticsearch > elasticsearch.log &

#### Run the example ####
python -m easy_elasticsearch.examples.quora --mode existing
