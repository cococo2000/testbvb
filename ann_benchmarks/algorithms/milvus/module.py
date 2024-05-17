""" Milvus CPU module with FLAT, IVFFLAT, IVFSQ8, IVFPQ, HNSW, SCANN index """
import subprocess
import numpy as np
from pymilvus import DataType, connections, utility, Collection, CollectionSchema, FieldSchema

from ann_benchmarks.algorithms.base.module import BaseANN


def metric_mapping(_metric: str):
    """
    Mapping metric type to milvus metric type

    Args:
        _metric (str): metric type
    
    Returns:
        str: milvus metric type
    """
    _metric = _metric.lower()
    _metric_type = {"angular": "COSINE", "euclidean": "L2"}.get(_metric, None)
    if _metric_type is None:
        raise ValueError(f"[Milvus] Not support metric type: {_metric}!!!")
    return _metric_type


class Milvus(BaseANN):
    """
    Milvus CPU module
    """
    def __init__(
            self,
            metric : str,
            dim : int
            ):
        self._metric = metric
        self._dim = dim
        self._metric_type = metric_mapping(self._metric)
        self.start_milvus()
        self.connects = connections
        self.connects.connect("default", host='localhost', port='19530', timeout=30)
        print(f"[Milvus] Milvus version: {utility.get_server_version()}")
        self.collection_name = "test_milvus"
        self.collection = None
        self.num_labels = 0
        self.search_params = {
            "metric_type": self._metric_type
        }
        self.name = f"Milvus metric:{self._metric}"
        if utility.has_collection(self.collection_name):
            print(f"[Milvus] collection {self.collection_name} already exists, drop it...")
            utility.drop_collection(self.collection_name)

    def start_milvus(self) -> None:
        """
        Start milvus cpu standalone docker compose
        """
        try:
            subprocess.run(["docker", "compose", "down"], check=True)
            subprocess.run(["docker", "compose", "up", "-d"], check=True)
            print("[Milvus] docker compose up successfully!!!")
        except subprocess.CalledProcessError as e:
            print(f"[Milvus] docker compose up failed: {e}!!!")

    def stop_milvus(self) -> None:
        """
        Stop milvus cpu standalone docker compose
        """
        try:
            subprocess.run(["docker", "compose", "down"], check=True)
            print("[Milvus] docker compose down successfully!!!")
        except subprocess.CalledProcessError as e:
            print(f"[Milvus] docker compose down failed: {e}!!!")

    def create_collection(
            self,
            num_labels : int = 0,
            label_names : list[str] | None = None,
            label_types : list[str] | None = None
            ) -> None:
        """
        Create collection with schema
        Args:
            num_labels (int): number of labels
            label_names (list[str]): label names
            label_types (list[str]): label types
        """
        filed_id = FieldSchema(
            name="id",
            dtype=DataType.INT64,
            is_primary=True
        )
        filed_vec = FieldSchema(
            name="vector",
            dtype=DataType.FLOAT_VECTOR,
            dim=self._dim
        )
        fields = [filed_id, filed_vec]
        self.num_labels = num_labels
        if num_labels > 0:
            label_type_to_dtype = {
                "BOOL": DataType.BOOL,
                "INT8": DataType.INT8,
                "INT16": DataType.INT16,
                "INT32": DataType.INT32,
                "INT64": DataType.INT64,
                "FLOAT": DataType.FLOAT,
                "DOUBLE": DataType.DOUBLE,
                "STRING": DataType.STRING,
                "VARCHAR": DataType.VARCHAR,
                "ARRAY": DataType.ARRAY,
                "JSON": DataType.JSON,
                "BINARY_VECTOR": DataType.BINARY_VECTOR,
                "FLOAT_VECTOR": DataType.FLOAT_VECTOR,
                "FLOAT16_VECTOR": DataType.FLOAT16_VECTOR,
                "BFLOAT16_VECTOR": DataType.BFLOAT16_VECTOR,
                "SPARSE_FLOAT_VECTOR": DataType.SPARSE_FLOAT_VECTOR,
                "UNKNOWN": DataType.UNKNOWN,
            }
            for i in range(num_labels):
                fields.append(
                    FieldSchema(
                        name=label_names[i],
                        dtype=label_type_to_dtype.get(label_types[i].upper(), DataType.UNKNOWN)
                    )
                )
        schema = CollectionSchema(
            fields = fields,
            description = "Test milvus search",
        )
        self.collection = Collection(
            self.collection_name,
            schema,
            consistence_level="STRONG"
        )
        print(f"[Milvus] Create collection {self.collection.describe()} successfully!!!")

    def insert(
            self,
            embeddings : np.ndarray,
            labels : np.ndarray | None = None
            ) -> None:
        """
        Insert embeddings and labels into collection

        Args:
            embeddings (np.ndarray): embeddings
            labels (np.ndarray): labels
        """
        batch_size = 1000
        if labels is not None:
            num_labels = len(labels[0])
            print(f"[Milvus] Insert {len(embeddings)} data with {num_labels} labels \
                  into collection {self.collection_name}...")
        else:
            print(f"[Milvus] Insert {len(embeddings)} data \
                  into collection {self.collection_name}...")
        for i in range(0, len(embeddings), batch_size):
            batch_data = embeddings[i : min(i + batch_size, len(embeddings))]
            entities = [
                [i for i in range(i, min(i + batch_size, len(embeddings)))],
                batch_data.tolist()
                ]
            if labels is not None:
                batch_labels = labels[i : min(i + batch_size, len(embeddings))]
                for j in range(num_labels):
                    entities.append(
                        [l[j] for l in batch_labels]
                    )
            self.collection.insert(entities)
        self.collection.flush()
        print(f"[Milvus] {self.collection.num_entities} data has been inserted into collection {self.collection_name}!!!")

    def get_index_param(self) -> dict:
        """
        Get index parameters

        Note: This is a placeholder method to be implemented by subclasses.
        """
        raise NotImplementedError()

    def create_index(self) -> None:
        """
        Create index for collection
        """
        print(f"[Milvus] Create index for collection {self.collection_name}...")
        self.collection.create_index(
            field_name = "vector",
            index_params = self.get_index_param(),
            index_name = "vector_index"
        )
        utility.wait_for_index_building_complete(
            collection_name = self.collection_name,
            index_name = "vector_index"
        )
        index = self.collection.index(index_name = "vector_index")
        index_progress =  utility.index_building_progress(
            collection_name = self.collection_name,
            index_name = "vector_index"
        )
        print(f"[Milvus] Create index {index.to_dict()} {index_progress} for collection {self.collection_name} successfully!!!")

    def load_collection(self) -> None:
        """
        Load collection
        """
        print(f"[Milvus] Load collection {self.collection_name}...")
        self.collection.load()
        utility.wait_for_loading_complete(self.collection_name)
        print(f"[Milvus] Load collection {self.collection_name} successfully!!!")

    def fit(
            self,
            embeddings : np.array,
            labels : np.ndarray | None = None,
            label_names : list[str] | None = None,
            label_types : list[str] | None = None
            ) -> None:
        """
        Fit the ANN algorithm to the provided data

        Args:
            embeddings (np.array): embeddings
            labels (np.array): labels
            label_names (list[str]): label names
            label_types (list[str]): label types
        """
        if labels is not None:
            self.create_collection(len(labels[0]), label_names, label_types)
        else:
            self.create_collection()
        self.insert(embeddings, labels)
        self.create_index()
        self.load_collection()

    def query(
            self,
            v : np.array,
            n : int,
            expr : str | None = None
            ) -> list[int]:
        """
        Performs a query on the algorithm to find the nearest neighbors

        Args:
            v (np.array): The vector to find the nearest neighbors of.
            n (int): The number of nearest neighbors to return.
            expr (str): The search expression
        
        Returns:
            list[int]: An array of indices representing the nearest neighbors.
        """
        results = self.collection.search(
            data = [v],
            anns_field = "vector",
            param = self.search_params,
            expr = expr,
            limit = n,
            output_fields=["id"]
        )
        ids = [r.entity.get("id") for r in results[0]]
        return ids

    def done(self) -> None:
        """
        Release resources
        """
        self.collection.release()
        utility.drop_collection(self.collection_name)
        self.stop_milvus()


class MilvusFLAT(Milvus):
    """ Milvus with FLAT index"""
    def __init__(
            self,
            metric : str,
            dim : int
            ):
        super().__init__(metric, dim)
        self.name = f"MilvusFLAT metric:{self._metric}"

    def get_index_param(self):
        return {
            "index_type": "FLAT",
            "metric_type": self._metric_type
        }

    def query(
            self,
            v : np.ndarray,
            n : int,
            expr : str | None = None
            ) -> list[int]:
        self.search_params = {
            "metric_type": self._metric_type,
        }
        results = self.collection.search(
            data = [v],
            anns_field = "vector",
            param = self.search_params,
            expr = expr,
            limit = n,
            output_fields=["id"]
        )
        ids = [r.entity.get("id") for r in results[0]]
        return ids


class MilvusIVFFLAT(Milvus):
    """ Milvus with IVF_FLAT index """
    def __init__(
            self,
            metric : str,
            dim : int,
            index_param: dict
            ):
        super().__init__(metric, dim)
        self._index_nlist = index_param.get("nlist", None)

    def get_index_param(self):
        return {
            "index_type": "IVF_FLAT",
            "params": {
                "nlist": self._index_nlist
            },
            "metric_type": self._metric_type
        }

    def set_query_arguments(
            self,
            nprobe : int
        ) -> None:
        """
        Set query arguments for IVF_FLAT index

        Args:
            nprobe (int): nprobe
        """
        self.search_params = {
            "metric_type": self._metric_type,
            "params": {"nprobe": nprobe}
        }
        self.name = f"MilvusIVFFLAT metric:{self._metric}, index_nlist:{self._index_nlist}, search_nprobe:{nprobe}"


class MilvusIVFSQ8(Milvus):
    """ Milvus with IVF_SQ8 index """
    def __init__(
            self,
            metric : str,
            dim : int,
            index_param: dict
            ):
        super().__init__(metric, dim)
        self._index_nlist = index_param.get("nlist", None)

    def get_index_param(self):
        return {
            "index_type": "IVF_SQ8",
            "params": {
                "nlist": self._index_nlist
            },
            "metric_type": self._metric_type
        }

    def set_query_arguments(
            self,
            nprobe : int
        ) -> None:
        """
        Set query arguments for IVF_SQ8 index

        Args:
            nprobe (int): nprobe
        """
        self.search_params = {
            "metric_type": self._metric_type,
            "params": {"nprobe": nprobe}
        }
        self.name = f"MilvusIVFSQ8 metric:{self._metric}, index_nlist:{self._index_nlist}, search_nprobe:{nprobe}"


class MilvusIVFPQ(Milvus):
    """ Milvus with IVF_PQ index """
    def __init__(
            self,
            metric : str,
            dim : int,
            index_param: dict
            ):
        super().__init__(metric, dim)
        self._index_nlist = index_param.get("nlist", None)
        self._index_m = index_param.get("m", None)
        self._index_nbits = index_param.get("nbits", None)

    def get_index_param(self):
        assert self._dim % self._index_m == 0, "dimension must be able to be divided by m"
        return {
            "index_type": "IVF_PQ",
            "params": {
                "nlist": self._index_nlist,
                "m": self._index_m,
                "nbits": self._index_nbits if self._index_nbits else 8 
            },
            "metric_type": self._metric_type
        }

    def set_query_arguments(
            self,
            nprobe : int
            ) -> None:
        """
        Set query arguments for IVF_PQ index

        Args:
            nprobe (int): nprobe
        """
        self.search_params = {
            "metric_type": self._metric_type,
            "params": {"nprobe": nprobe}
        }
        self.name = f"MilvusIVFPQ metric:{self._metric}, \
            index_nlist:{self._index_nlist}, search_nprobe:{nprobe}"


class MilvusHNSW(Milvus):
    """ Milvus with HNSW index """
    def __init__(
            self,
            metric : str,
            dim : int,
            index_param: dict
            ):
        super().__init__(metric, dim)
        self._index_m = index_param.get("M", None)
        self._index_ef = index_param.get("efConstruction", None)

    def get_index_param(self):
        return {
            "index_type": "HNSW",
            "params": {
                "M": self._index_m,
                "efConstruction": self._index_ef
            },
            "metric_type": self._metric_type
        }

    def set_query_arguments(
            self,
            ef : int
            ) -> None:
        """
        Set query arguments for HNSW index

        Args:
            ef (int): ef
        """
        self.search_params = {
            "metric_type": self._metric_type,
            "params": {"ef": ef}
        }
        self.name = f"MilvusHNSW metric:{self._metric}, index_M:{self._index_m}, index_ef:{self._index_ef}, search_ef={ef}"


class MilvusSCANN(Milvus):
    """ Milvus with SCANN index """
    def __init__(
            self,
            metric : str,
            dim : int,
            index_param: dict
            ):
        super().__init__(metric, dim)
        self._index_nlist = index_param.get("nlist", None)

    def get_index_param(self):
        return {
            "index_type": "SCANN",
            "params": {
                "nlist": self._index_nlist
            },
            "metric_type": self._metric_type
        }

    def set_query_arguments(
            self,
            nprobe : int
        ) -> None:
        """
        Set query arguments for IVF_SQ8 index

        Args:
            nprobe (int): nprobe
        """
        self.search_params = {
            "metric_type": self._metric_type,
            "params": {"nprobe": nprobe}
        }
        self.name = f"MilvusSCANN metric:{self._metric}, index_nlist:{self._index_nlist}, search_nprobe:{nprobe}"
