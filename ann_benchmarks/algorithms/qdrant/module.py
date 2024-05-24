""" Qdrant module for ANN_Benchmarks framework. """
import subprocess
from time import sleep
from typing import List
import docker
import numpy as np

from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition
from qdrant_client.http.models import PointStruct, CollectionStatus

from ann_benchmarks.algorithms.base.module import BaseANN

def metric_mapping(_metric: str):
    """
    Mapping metric type to Qdrant distance metric

    Args:
        _metric (str): metric type

    Returns:
        str: Qdrant distance metric type
    """
    _metric = _metric.lower()
    _metric_type = {
        "dot": Distance.DOT,
        "angular": Distance.COSINE,
        "euclidean": Distance.EUCLID
    }.get(_metric, None)
    if _metric_type is None:
        raise ValueError(f"[Qdrant] Not support metric type: {_metric}!!!")
    return _metric_type


class Qdrant(BaseANN):
    def __init__(self, metric, m, ef_construct):
        self._metric = metric
        self._metric_type = metric_mapping(metric)
        self._collection_name = "qdrant_test"
        self._m = m
        self._ef_construct = ef_construct
        self.docker_client = None
        self.docker_name = "qdrant"
        self.container = None
        self.start_container()
        self.client = QdrantClient(url="http://localhost:6333", timeout=10)
        print("[qdrant] client connected successfully!!!")
        self.num_labels = 0
        if self.client.collection_exists(self._collection_name):
            print("[qdrant] collection already exists!!!")
            self.client.delete_collection(self._collection_name)
            print("[qdrant] collection deleted successfully!!!")
        self.name = f"Qdrant metric:{metric} m:{m} ef_construct:{ef_construct}"
        self.search_params = None

    def start_container(self) -> None:
        """
        Start qdrant by docker run
        """
        self.docker_client = docker.from_env()
        if self.docker_client.containers.list(filters={"name": self.docker_name}) != []:
            print("[qdrant] docker container already exists!!!")
            self.container = self.docker_client.containers.get(self.docker_name)
            self.stop_container()
        subprocess.run(["docker", "pull", "qdrant/qdrant:v1.9.2"], check=True)
        self.container = self.docker_client.containers.run(
            "qdrant/qdrant:v1.9.2",
            name=self.docker_name,
            volumes={
                "/tmp/qdrant_storage": {
                    "bind": "/qdrant/storage",
                    "mode": "z"
                }
            },
            ports={
                "6333/tcp": 6333,
                "6334/tcp": 6334
            },
            detach=True
        )
        print("[qdrant] docker start successfully!!!")
        sleep(10)

    def stop_container(self) -> None:
        """
        Stop qdrant
        """
        self.container.stop()
        self.container.remove(force=True)
        print("[qdrant] docker stop successfully!!!")

    def load_data(
            self,
            embeddings: np.array,
            labels: np.ndarray | None = None,
            label_names: list[str] | None = None,
            label_types: list[str] | None = None,
            ) -> None:
        dimensions = embeddings.shape[1]
        num_labels = len(label_names) if label_names is not None else 0
        self.num_labels = num_labels
        print(f"[qdrant] load data with {num_labels} labels!!!")
        self.client.create_collection(
            collection_name=self._collection_name,
            vectors_config=VectorParams(
                size=dimensions,
                distance=self._metric_type,
                hnsw_config=models.HnswConfigDiff(
                    m=self._m,
                    ef_construct=self._ef_construct,
                ),
                quantization_config=models.ProductQuantization(
                    product=models.ProductQuantizationConfig(
                        compression=models.CompressionRatio.X32,
                        always_ram=True
                    )
                )
            ),
        )
        print("[qdrant] collection created successfully!!!")
        print(f"[qdrant] Start uploading {len(embeddings)} vectors!!!")
        batch_size = 1000
        for i in range(0, len(embeddings), batch_size):
            points = []
            for j in range(i, min(i + batch_size, len(embeddings))):
                payload = {}
                if num_labels > 0:
                    for k in range(num_labels):
                        payload[label_names[k]] = int(labels[j][k])
                points.append(PointStruct(
                    id=j,
                    vector=embeddings[j],
                    payload=payload
                ))
            self.client.upsert(
                collection_name=self._collection_name,
                points=points,
                wait=True
            )
        print(f"[qdrant] Uploaded {len(embeddings)} vectors successfully!!!")
        # wait for vectors to be fully indexed
        while True:
            sleep(5)
            collection_info = self.client.get_collection(self._collection_name)
            if collection_info.status != CollectionStatus.GREEN:
                continue
            else:
                print(f"[qdrant] Point count: {collection_info.points_count}")
                print(f"[qdrant] Stored vectors: {collection_info.vectors_count}")
                print(f"[qdrant] Indexed vectors: {collection_info.indexed_vectors_count}")
                print(f"[qdrant] Collection status: {collection_info.status}")
                break

    def create_index(self):
        """ Qdrant has already created index during data upload """
        return

    def set_query_arguments(self, ef, exact):
        """
        Set query arguments for weaviate query with hnsw index
        """
        self.search_params = models.SearchParams(hnsw_ef=ef, exact=exact)
        self.name = f"Qdrant metric:{self._metric} m:{self._m} ef_construct:{self._ef_construct} ef:{ef} exact:{exact}"

    def convert_expr_to_filter(self, expr: str):
        """
        Convert a filter expression to a Filter object list

        Args:
            expr (str): filter expression. Example: "age > 20 and height < 180 or weight == 70"

        Returns:
            Filter: Filter object for qdrant query
        """
        tokens = expr.split()
        must_filters = []
        must_not_filters = []

        i = 1
        while i < len(tokens):
            if tokens[i] == "and":
                i += 2
            elif tokens[i] == "or":
                raise ValueError(f"[qdrant] we have not supported 'or' operator in expression!!!, expr: {expr}")
            elif tokens[i] in ["==", ">=", "<=", ">", "<", "!="]:
                left = tokens[i - 1]
                operator = tokens[i]
                right = tokens[i + 1]
                # print(f"[qdrant] left: {left}, operator: {operator}, right: {right}")
                i += 4
                if operator == ">=":
                    must_filters.append(FieldCondition(key=left, range=models.Range(gte=int(right))))
                elif operator == "<=":
                    must_filters.append(FieldCondition(key=left, range=models.Range(lte=int(right))))
                elif operator == ">":
                    must_filters.append(FieldCondition(key=left, range=models.Range(gt=int(right))))
                elif operator == "<":
                    must_filters.append(FieldCondition(key=left, range=models.Range(lt=int(right))))
                elif operator == "==":
                    must_filters.append(FieldCondition(key=left, match=models.MatchValue(value=int(right))))
                elif operator == "!=":
                    must_not_filters.append(FieldCondition(key=left, match=models.MatchValue(value=int(right))))
            else:
                raise ValueError(f"[qdrant] Unsupported operator: {tokens[i]}")
        return must_filters, must_not_filters

    def query(self, v, n, expr=None):
        must_filters, must_not_filters = self.convert_expr_to_filter(expr)
        # print(f"[qdrant] must_filters: {must_filters}")
        # print(f"[qdrant] must_not_filters: {must_not_filters}")
        ret = self.client.search(
            collection_name=self._collection_name,
            query_vector=v,
            query_filter=Filter(
                must = must_filters,
                must_not = must_not_filters
            ),
            search_params=self.search_params,
            limit=n,
        )
        return [point.id for point in ret]

    def done(self):
        self.client.delete_collection(self._collection_name)
        print("[qdrant] collection deleted successfully!!!")
        self.client.close()
        self.stop_container()
