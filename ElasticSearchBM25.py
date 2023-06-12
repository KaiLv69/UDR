import os
from elasticsearch import Elasticsearch, helpers, NotFoundError
import time
import tqdm
import requests
import os
import time
import requests
import tarfile
import subprocess
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)
logging.getLogger("elasticsearch").setLevel(logging.CRITICAL)  # muting logging from ES
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO
)


class ElasticSearchBM25(object):
    """
    Connect to the Elasticsearch service when both valid `host` and `port_http` indicated or create a new one via docker when `host` is None.
    :param corpus: A mapping from IDs to docs.
    :param index_name: Name of the elasticsearch index.
    :param reindexing: Whether to re-index the documents if the index exists.
    :param port_http: The HTTP port of the elasticsearch service.
    :param port_tcp: The TCP port of the elasticsearch service.
    :param host: The host address of the elasticsearch service. If set None, an ES docker container will be started with the indicated port numbers, `port_http` and `port_tcp` exposed.
    :param service_type: When starting ES service needed, use either "docker" to start a new ES docker container or "executable" to download executable ES and run.
    :param es_version: Indicating the elasticsearch version for the docker container.
    :param timeout: Timeout (in seconds) at the ES-service side.
    :param max_waiting: Maximum time (in seconds) to wait for starting the elasticsearch docker container.
    :param cache_dir: Cache directory for downloading the ES executable if needed.
    """

    def __init__(
        self,
        corpus: Dict[str, str],
        index_name: str = "one_trial",
        reindexing: bool = True,
        port_http: str = "9200",
        port_tcp: str = "9300",
        host: str = None,
        service_type: str = "docker",
        es_version: str = "7.9.1",
        timeout: int = 100,
        max_waiting: int = 100,
        cache_dir: str = "/tmp",
    ):
        self.container_name = None
        self.pid = None
        # if host is not None:
        #     self._wait_and_check(host, port_http, max_waiting)
        #     logger.info(f"Successfully reached out to ES service at {host}:{port_http}")
        # else:
        #     host = "http://localhost"
        #     if self._check_service_running(host, port_http):
        #         logger.info(
        #             f"Successfully reached out to ES service at {host}:{port_http}"
        #         )
        #     elif service_type == "docker":
        #         logger.info("No host running. Now start a new ES service via docker")
        #         self.container_name = self._start_docker_service(
        #             port_http, port_tcp, es_version, max_waiting
        #         )
        #     elif service_type == "executable":
        #         logger.info(
        #             "No host running. Now start a new ES service via downloading the executable software"
        #         )
        #         self.pid = self._start_executable_service(
        #             port_http, port_tcp, es_version, max_waiting, cache_dir
        #         )

        es = Elasticsearch(
            [
                {"host": "localhost", "port": port_http},
            ],
            timeout=timeout,
        )
        logger.info(
            f"Successfully built connection to ES service at {host}:{port_http}"
        )
        self.es = es
        if es.indices.exists(index=index_name):
            if reindexing:
                logger.info(
                    f"Index {index_name} found and it will be indexed again since reindexing=True"
                )
                es.indices.delete(index=index_name)
                self._index_corpus(corpus, index_name)
        else:
            logger.info(f"No index found and now do indexing")
            self._index_corpus(corpus, index_name)
        self.index_name = index_name
        logger.info("All set up.")

    def _check_service_running(self, host, port) -> bool:
        """
        Check whether the ES service is reachable.
        :param host: The host address.
        :param port: The HTTP port.
        :return: Whether the ES service is reachable.
        """
        try:
            return requests.get(f"{host}:{port}").status_code == 200
        except:
            return False

    def _wait_and_check(self, host, port, max_waiting) -> bool:
        logger.info(
            f"Waiting for the ES service to be well started. Maximum time waiting: {max_waiting}s"
        )
        timeout = True
        for _ in tqdm.trange(max_waiting):
            if self._check_service_running(host, port):
                timeout = False
                break
            time.sleep(1)
        assert timeout == False, (
            "Timeout to start the ES docker container or connect to the ES service, "
            + "please increase max_waiting or check the idling ES services "
            + "(starting multiple ES instances from ES executable is not allowed)"
        )

    def _start_docker_service(self, port_http, port_tcp, es_version, max_waiting):
        """
        Start an ES docker container at localhost.
        :param port_http: The HTTP port.
        :param port_tcp: The TCP port.
        :param es_version: The ES version.
        :param max_waiting: Maximum time of waiting for starting the docker container.
        :return: Name of the docker container.
        """
        host = "http://localhost"
        assert (
            os.system("docker") == 0
        ), "Cannot run docker! Please make sure docker has been installed correctly."
        container_name = f"easy-elasticsearch-node{int(time.time())}"
        cmd = (
            f'docker run -p {port_http}:9200 -p {port_tcp}:9300 -e "discovery.type=single-node" --detach '
            + f"--name {container_name} docker.elastic.co/elasticsearch/elasticsearch:{es_version}"
        )
        logger.info(f"Running command: `{cmd}`")
        os.system(cmd)
        self._wait_and_check(host, port_http, max_waiting)
        logger.info(f'Successfully started a ES container with name "{container_name}"')
        return container_name

    def _start_executable_service(
        self, port_http, port_tcp, es_version, max_waiting, cache_dir
    ):
        """
        Start an ES service from an executable program at localhost.
        :param port_http: The HTTP port.
        :param port_tcp: The TCP port.
        :param es_version: The ES version.
        :param max_waiting: Maximum time of waiting for starting the docker container.
        :return: Name of the docker container.
        """
        host = "http://localhost"
        download_path = cache_dir
        excutable_path = os.path.join(
            download_path, f"elasticsearch-{es_version}", "bin", "elasticsearch"
        )

        if not os.path.exists(excutable_path):
            tar_gz = os.path.join(
                download_path, f"elasticsearch-{es_version}-linux-x86_64.tar.gz"
            )
            logger.info(f"Found no existing executable. Now download it to {tar_gz}")
            chunk_size = 1024
            url = f"https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-{es_version}-linux-x86_64.tar.gz"
            r = requests.get(url, stream=True)
            total = int(r.headers.get("Content-Length", 0))
            logger.info("Begin downloading the ES excutable")
            with open(tar_gz, "wb") as f:
                progress_bar = tqdm.tqdm(
                    total=total, unit="iB", unit_scale=True, unit_divisor=chunk_size
                )
                for data in r.iter_content(chunk_size=chunk_size):
                    size = f.write(data)
                    progress_bar.update(size)

            logger.info("Unzipping the tar.gz file")
            with tarfile.open(tar_gz, "r:gz") as f:
                f.extractall(path=download_path)

        logger.info("Starting ES")
        assert os.path.exists(excutable_path), "Cannot find the executable!"
        cmd = f"{excutable_path} -E http.port={port_http} -E transport.tcp.port={port_tcp}"
        logger.info(f"Running command `{cmd}`")
        with open(f"es-{es_version}.log", "w") as fstdout:
            proc = subprocess.Popen(
                cmd, shell=True, stdout=fstdout, stderr=subprocess.STDOUT
            )
            es_pid = proc.pid + 1
            logger.info(f"PID: {es_pid}")

        self._wait_and_check(host, port_http, max_waiting)
        logger.info(f"Successfully started a ES service from executable")
        return es_pid

    def _index_corpus(self, corpus, index_name):
        """
        Index the corpus.
        :param corpus: A mapping from document ID to documents.
        :param index_name: The name of the target ES index.
        """
        es_index = {
            "mappings": {
                "properties": {
                    "document": {"type": "text"},
                }
            }
        }
        self.es.indices.create(index=index_name, body=es_index, ignore=[400])
        ndocuments = len(corpus)
        dids, documents = list(corpus.keys()), list(corpus.values())
        chunk_size = 500
        pbar = tqdm.trange(0, ndocuments, chunk_size)
        for begin in pbar:
            did_chunk = dids[begin : begin + chunk_size]
            document_chunk = documents[begin : begin + chunk_size]
            bulk_data = [
                {
                    "_index": index_name,
                    "_id": did,
                    "_source": {
                        "document": documnt,
                    },
                }
                for did, documnt in zip(did_chunk, document_chunk)
            ]
            helpers.bulk(self.es, bulk_data)
        self.es.indices.refresh(
            index=index_name
        )  # important!!! otherwise es might return nothing!!!
        logger.info(f"Indexing work done: {ndocuments} documents indexed")

    def query(self, query: str, topk, return_scores=False) -> Dict[str, str]:
        """
        Search for a given query.
        :param query: The query text.
        :param topk: Specifying how many top documents to return. Should less than 10000.
        :param return_scores: Whether to return the scores.
        :return: Ranked documents, a mapping from IDs to the documents (and also the scores, a mapping from IDs to scores).
        """
        assert topk <= 10000, "`topk` is too large!"
        result = self.es.search(
            index=self.index_name,
            size=min(topk, 10000),
            body={"query": {"match": {"document": query}}},
        )
        hits = result["hits"]["hits"]
        documents_ranked = {hit["_id"]: hit["_source"]["document"] for hit in hits}
        if return_scores:
            scores_ranked = {hit["_id"]: hit["_score"] for hit in hits}
            return documents_ranked, scores_ranked
        else:
            return documents_ranked

    def score(
        self, query: str, document_ids: List[int], max_ntries=60
    ) -> Dict[str, str]:
        """
        Scoring a query against the given documents (IDs).
        :param query: The query text.
        :param document_ids: The document IDs.
        :param max_ntries: Maximum time (in seconds) for trying.
        :return: The mapping from IDs to scores.
        """
        for i in range(max_ntries):
            try:
                scores = {}
                for document_id in document_ids:
                    result = self.es.explain(
                        index=self.index_name,
                        id=document_id,
                        body={"query": {"match": {"document": query}}},
                    )
                    scores[document_id] = result["explanation"]["value"]
                return scores
            except NotFoundError as e:
                if i == max_ntries:
                    raise e
                logger.info(f"NotFoundError, now re-trying ({i+1}/{max_ntries}).")
                time.sleep(1)

    def delete_index(self):
        """
        Delete the used index.
        """
        if self.es.indices.exists(index=self.index_name):
            logger.info(
                f'Delete "{self.index_name}": {self.es.indices.delete(self.index_name)}'
            )
        else:
            logger.warning(f'Index "{self.index_name}" does not exist!')

    def delete_container(self):
        """
        Delete the used docker container.
        """
        if self.container_name is not None:
            cmd = f"docker rm -f {self.container_name}"
            logger.info(f'Delete container "{self.container_name}": {os.system(cmd)}')
        else:
            logger.warning(f"No running ES container found!")
            cmd = 'docker ps | grep "easy-elasticsearch-node"'
            with os.popen(cmd) as f:
                idling_nodes = f.read()
            if idling_nodes:
                logger.warning(f"Found idling nodes:\n {idling_nodes}.")

    def delete_excutable(self):
        """
        Kill the ES process
        """
        if self.pid is not None:
            logger.info(f"Kill process of PID {self.pid}")
            os.kill(self.pid, 15)
        else:
            logger.warning(f"No running ES service found!")
            cmd = "ps -ef | grep elasticsearch"
            with os.popen(cmd) as f:
                idling_nodes = f.read()
            if idling_nodes:
                logger.warning(f"Found idling process:\n {idling_nodes}.")
