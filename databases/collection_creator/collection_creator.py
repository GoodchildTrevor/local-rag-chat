import os
import sys

from qdrant_client.models import models

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from config.settings import AppConfig, ClientsConfig, EmbeddingModelsConfig

app_config=AppConfig()
client_config = ClientsConfig()
embedding_models_config = EmbeddingModelsConfig()


class CreateCollection:
    """
    Utility class for creating or recreating a Qdrant collection
    with dense, sparse, and optionally late-interaction embeddings.
    :param collection_name: Name of the collection to create or recreate.
    :param dense_embeddings: List of dense embedding vectors.
    :param late_embeddings: Optional 3D list of late interaction vectors for multi-vector configuration.
    :param late: Flag to include late interaction vector configuration.
    :param recreation: Whether to recreate the collection (drops and creates anew).
    """

    def __init__(
        self,
        app_config: AppConfig,
        client_config: ClientsConfig,
        collection_name: str,
        dense_embeddings: list[list[float]],
        late_embeddings: list[list[list[float]]] = None,
        sparse: bool = True,
        late: bool = False,
        recreation: bool = False
    ):
        self.app_config = app_config
        self.client = client_config.qdrant_client
        self.collection_name = collection_name
        self.dense_embeddings = dense_embeddings
        self.late_embeddings = late_embeddings
        self.sparse = sparse
        self.late = late
        self.recreation = recreation

    def creator(self, configs: dict) -> None:
        """
        Creates a Qdrant collection with the given configuration.
        :param configs: Dictionary containing vector and sparse vector configurations.
        """
        if self.sparse:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=configs["vectors_config"],
                sparse_vectors_config=configs["sparse_vectors_config"],
            )
        else:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=configs["vectors_config"],
            )

    def recreator(self, configs: dict) -> None:
        """
        Recreates (deletes and creates) a Qdrant collection with the given configuration.
        :param configs: Dictionary containing vector and sparse vector configurations.
        """
        if self.sparse:
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=configs["vectors_config"],
                sparse_vectors_config=configs["sparse_vectors_config"],
            )
        else:
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=configs["vectors_config"],
            )

    def build_collection(self,) -> None:
        """
        Builds the Qdrant collection based on the provided configuration flags and embeddings.
        Uses cosine distance for all vector spaces. Supports optional late-interaction vectors
        using `MultiVectorConfig` with `MAX_SIM` comparator.
        """
        dense_config = {
            app_config.dense_vector_config: models.VectorParams(
                size=len(self.dense_embeddings[0]),
                distance=models.Distance.COSINE,
            ),
        }
        sparse_config = {
            app_config.sparse_vector_config: models.SparseVectorParams(
                modifier=models.Modifier.IDF
            ),
        }
        late_config = {
            app_config.late_vector_config: models.VectorParams(
                size=len(self.late_embeddings[0][0]),
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM,
                ),
            )
        }

        if self.late:
            vectors_config = {**dense_config, **late_config}
        else:
            vectors_config = dense_config
        
        configs = {
            "vectors_config": vectors_config,
        }
        if self.sparse:
            configs["sparse_vectors_config"] = sparse_config

        if self.recreation:
            self.recreator(configs)
        else:
            self.creator(configs)


if __name__ == "__main__":
    sample_text = "Hello World"
    dense_embeddings = list(embedding_models_config.dense.embed(sample_text))
    late_embeddings = [list(vectors) for vectors in embedding_models_config.late.embed(sample_text)]
    CreateCollection(
        app_config=app_config,
        client_config=client_config.qdrant_client,
        collection_name=app_config.rag_collection,
        dense_embeddings=dense_embeddings,
        late_embeddings=late_embeddings,
    ).build_collection()
