# Create ES container

## Step 1: Download Elastic docker image (if it is installed in the system, move to step 2)

```bash
docker network create elastic
docker pull docker.elastic.co/elasticsearch/elasticsearch:9.2.3
```

## Step 2: Start container
```bash
docker run --name elastic-rag-eval \
  --net elastic \
  -p 9202:9200 \
  -d \
  -m 6GB \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=true" \
  -e "ES_JAVA_OPTS=-Xms3g -Xmx3g" \
  -v es_data_eval:/usr/share/elasticsearch/data \
  docker.elastic.co/elasticsearch/elasticsearch:9.2.3
```
Careful: the first port in the `-p` argument must be different for the port uses by another instances (see `docker ps -a` to check current instances). Be careful also with the data volume (`-v`argument). If there is any problem, change the name to de data variable (`es_data_eval` in this example).

## Step 3: Set password
```bash
docker exec -it elastic-rag-eval /usr/share/elasticsearch/bin/elasticsearch-reset-password -u elastic
```
Add this generated password in the `config_elastic.yaml` file