""" Weaviate implementation for the ANN-Benchmarks framework. """
import subprocess
import uuid
import numpy as np
import time

import weaviate
import weaviate.classes as wvc
from weaviate.classes.config import Property, DataType, Configure, VectorDistances, Reconfigure
from weaviate.classes.query import Filter

from ann_benchmarks.algorithms.base.module import BaseANN


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
    Weaviate module for the ANN-Benchmarks framework
    """
    def __init__(
            self,
            metric : str,
            max_connections,
            ef_construction=512
        ):
        self._metric = metric
        self._metric_type = metric_mapping(metric)
        self.max_connections = max_connections
        self.ef_construction = ef_construction
        self.start_weaviate()
        time.sleep(10)
        max_tries = 10
        for i in range(max_tries):
            try:
                self.client = weaviate.connect_to_local()
                # self.client = weaviate.
                break
            except Exception as e:
                print(f"[weaviate] connection failed: {e}")
                time.sleep(1)
        self.collection_name = "test_weaviate"
        self.collection = None
        self.num_labels = 0
        self.name = f"Weaviate metric={metric}"
        if self.client.collections.exists(self.collection_name):
            self.client.collections.delete(self.collection_name)

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
        self.client.collections.create(
            self.collection_name,
            properties=properties,
            # vector_index_config=Configure.VectorIndex.hnsw(
            #     distance_metric=self._metric_type,
            #     ef_construction=self.ef_construction,
            #     max_connections=self.max_connections,
            # ),
            inverted_index_config=Configure.inverted_index(  # Optional
                bm25_b=0.7,
                bm25_k1=1.25,
                index_null_state=True,
                index_property_length=True,
                index_timestamps=True
            ),
        )
        self.collection = self.client.collections.get(self.collection_name)
        batch_size = 1000
        for i in range(0, len(embeddings), batch_size):
            data_objects = []
            print(f"[weaviate] load data: {i}/{len(embeddings)}")
            for j in range(i, min(i + batch_size, len(embeddings))):
                properties = {}
                if num_labels > 0:
                    for k in range(num_labels):
                        # print(f"[weaviate] labels[j][k]: {labels[j][k]} {type(labels[j][k])}")
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

    def set_query_arguments(self, ef):
        self.collection.config.update(
            vectorizer_config=Reconfigure.VectorIndex.hnsw(
                ef=ef
            )
        )
        print(f"[weaviate] set_query_arguments: {ef}")
        print(f"[weaviate] Collection Config: {self.collection.config.get()}")

    def expr2filter(self, expr):
        if expr is None:
            return None
        if expr

    def query(self, v, n, expr=None):
        # print(f"[weaviate] query: {v}")
        # print(f"[weaviate] query: {n}")
        # print(f"[weaviate] query: {v.tolist()}")
        # filters = None
        filters = eval("""Filter.by_property("text_length").greater_or_equal(3) & Filter.by_property("text_length").less_or_equal(63)""")
        ret = self.collection.query.near_vector(
            near_vector=v.tolist(),
            limit=n,
            filters=filters,
        )
        # print(expr)
        # print(f"[weaviate] query: {ret}")

        # print(f"[weaviate] query: {ret}")
        # print("[weaviate] query: ", ret.total_count)
        ids = [int(o.uuid) for o in ret.objects]
        return ids

    def done(self):
        self.client.close()
        self.stop_weaviate()
