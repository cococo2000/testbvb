""" Weaviate implementation for the ANN-Benchmarks framework. """
import subprocess
import uuid
import time
import numpy as np

import weaviate
import weaviate.classes as wvc
from weaviate.classes.config import Property, DataType, Configure, VectorDistances, Reconfigure
from weaviate.classes.query import Filter

from ann_benchmarks.algorithms.base.module import BaseANN
from ann_benchmarks.algorithms.weaviate.utils import convert_conditions_to_filters


def metric_mapping(_metric: str):
    """
    Mapping metric type to weaviate metric type

    Args:
        _metric (str): metric type

    Returns:
        str: Weaviate metric type
    """
    _metric = _metric.lower()
    _metric_type = {
        "angular": VectorDistances.COSINE,
        "euclidean": VectorDistances.L2_SQUARED,
        "hamming" : VectorDistances.HAMMING
    }.get(_metric, None)
    if _metric_type is None:
        raise ValueError(f"[Weaviate] Not support metric type: {_metric}!!!")
    return _metric_type


class Weaviate(BaseANN):
    """
    Weaviate base module
    """
    def __init__(
            self,
            metric : str
        ):
        self._metric = metric
        self._metric_type = metric_mapping(metric)
        self.start_weaviate()
        time.sleep(10)
        max_tries = 10
        for _ in range(max_tries):
            try:
                self.client = weaviate.connect_to_local()
                break
            except Exception as e:
                print(f"[weaviate] connection failed: {e}")
                time.sleep(1)
        self.collection_name = "test_weaviate"
        self.collection = None
        self.num_labels = 0
        self.name = f"Weaviate metric:{metric}"
        if self.client.collections.exists(self.collection_name):
            self.client.collections.delete(self.collection_name)
        self.query_vector = None
        self.query_topk = 0
        self.query_filters = None
        self.prepare_query_results = None

    def start_weaviate(self) -> None:
        """
        Start weaviate by docker compose
        """
        try:
            subprocess.run(["docker", "compose", "down"], check=True)
            subprocess.run(["docker", "compose", "up", "-d"], check=True)
            print("[weaviate] docker compose up successfully!!!")
        except subprocess.CalledProcessError as e:
            print(f"[weaviate] docker compose up failed: {e}!!!")

    def stop_weaviate(self) -> None:
        """
        Stop weaviate by docker compose
        """
        try:
            subprocess.run(["docker", "compose", "down"], check=True)
            print("[weaviate] docker compose down successfully!!!")
        except subprocess.CalledProcessError as e:
            print(f"[weaviate] docker compose down failed: {e}!!!")

    def create_collection(self, properties) -> None:
        """
        Create collection with schema
        
        Args:
            properties (list): list of properties
        """
        raise NotImplementedError

    def load_data(
            self,
            embeddings: np.array,
            labels: np.ndarray | None = None,
            label_names: list[str] | None = None,
            label_types: list[str] | None = None,
            ) -> None:
        num_labels = len(label_names) if label_names is not None else 0
        self.num_labels = num_labels
        print(f"[weaviate] num_labels: {num_labels}")
        # Create a collection and define properties
        properties = []
        if num_labels > 0:
            label_type_to_weaviate_type = {
                "BOOL": DataType.BOOL,
                "INT": DataType.INT,
                "INT32": DataType.INT,
                "FLOAT": DataType.NUMBER,
                "STRING": DataType.TEXT
            }
            for label_name, label_type in zip(label_names, label_types):
                properties.append(
                    Property(
                        name=label_name,
                        data_type=label_type_to_weaviate_type[label_type.upper()]
                    )
                )
        self.create_collection(properties)
        self.collection = self.client.collections.get(self.collection_name)
        print(f"[weaviate] Start loading data with {len(embeddings)} data...")
        batch_size = 1000
        print(f"[weaviate] load data with batch size: {batch_size}")
        for i in range(0, len(embeddings), batch_size):
            data_objects = []
            for j in range(i, min(i + batch_size, len(embeddings))):
                properties = {}
                if num_labels > 0:
                    for k in range(num_labels):
                        # TODO: fix if the type of labels is not int/int32
                        properties[label_names[k]] = int(labels[j][k])
                data_objects.append(
                    wvc.data.DataObject(
                        uuid=uuid.UUID(int=j),
                        properties=properties,
                        vector=embeddings[j].tolist(),
                    )
                )
            self.collection.data.insert_many(data_objects)
        print(f"[weaviate] load {len(embeddings)} data successfully!!!")

    def create_index(self) -> None:
        # Weaviate has already created the index before loading the data
        pass

    def query(self, v, n, expr=None):
        if expr is not None:
            filters = eval(convert_conditions_to_filters(expr))
        else:
            filters = None
        ret = self.collection.query.near_vector(
            near_vector=v.tolist(),
            limit=n,
            filters=filters,
        )
        ids = [int(o.uuid) for o in ret.objects]
        return ids

    def prepare_query(
            self,
            v : np.array,
            n : int,
            expr : str | None = None
            ) -> None:
        """
        Prepare query

        Args:
            v (np.array): The vector to find the nearest neighbors of.
            n (int): The number of nearest neighbors to return.
            expr (str): The search expression
        """
        self.query_vector = v.tolist()
        self.query_topk = n
        self.query_filters = eval(convert_conditions_to_filters(expr)) if expr is not None else None

    def run_prepared_query(self) -> None:
        """
        Run prepared query
        """
        ret = self.collection.query.near_vector(
            near_vector=self.query_vector,
            limit=self.query_topk,
            filters=self.query_filters,
        )
        self.prepare_query_results = [int(o.uuid) for o in ret.objects]

    def get_prepared_query_results(self) -> list[int]:
        """
        Get prepared query results

        Returns:
            list[int]: An array of indices representing the nearest neighbors.
        """
        return self.prepare_query_results

    def done(self):
        self.client.close()
        self.stop_weaviate()


class WeaviateFLAT(Weaviate):
    """
    Weaviate with FLAT index
    """
    def __init__(
            self,
            metric : str,
        ):
        super().__init__(metric)
        self.name = f"WeaviateFLAT metric:{metric}"

    def create_collection(self, properties) -> None:
        self.client.collections.create(
            name=self.collection_name,
            properties=properties,
            vector_index_config=Configure.VectorIndex.flat(
                distance_metric=self._metric_type,
                quantizer=Configure.VectorIndex.Quantizer.bq()
            ),
            inverted_index_config=Configure.inverted_index()
        )


class WeaviateHNSW(Weaviate):
    """
    Weaviate with HNSW index
    """
    def __init__(
            self,
            metric : str,
            index_param: dict,
        ):
        super().__init__(metric)
        self.max_connections = index_param.get("M", None)
        self.ef_construction = index_param.get("efConstruction", None)

    def create_collection(self, properties) -> None:
        """
        Create collection with schema
        
        Args:
            properties (list): list of properties
        """
        self.client.collections.create(
            name=self.collection_name,
            properties=properties,
            vector_index_config=Configure.VectorIndex.hnsw(
                distance_metric=self._metric_type,
                ef_construction=self.ef_construction,
                max_connections=self.max_connections,
                quantizer=Configure.VectorIndex.Quantizer.pq()
            ),
            inverted_index_config=Configure.inverted_index()
        )

    def set_query_arguments(self, ef):
        """
        Set query arguments for weaviate query with hnsw index
        """
        self.collection.config.update(
            vectorizer_config=Reconfigure.VectorIndex.hnsw(
                ef=ef
            )
        )
        self.name = f"WeaviateHNSW metric:{self._metric} max_connections:{self.max_connections} ef_construction:{self.ef_construction} ef:{ef}"
        print(f"[weaviate] set_query_arguments: {ef}")
        print(f"[weaviate] Collection Config: {self.collection.config.get()}")
