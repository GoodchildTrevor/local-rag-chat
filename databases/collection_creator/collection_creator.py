from dotenv import load_dotenv
import os
import sys

from qdrant_client.models import models

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from config.settings import get_settings

load_dotenv()
collection = os.getenv("CASH_COLLECTION")

settings = get_settings()


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
        collection_name: str,
        dense_embeddings: list[list[float]],
        late_embeddings: list[list[list[float]]] = None,
        sparse: bool = True,
        late: bool = False,
        recreation: bool = False
    ):
        self.client = settings.client
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
            settings.dense_vector_config: models.VectorParams(
                size=len(self.dense_embeddings[0]),
                distance=models.Distance.COSINE,
            ),
        }
        sparse_config = {
            settings.sparse_vector_config: models.SparseVectorParams(
                modifier=models.Modifier.IDF
            ),
        }
        late_config = {
            settings.late_vector_config: models.VectorParams(
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
    dense_embeddings = list(settings.dense_embedding_model.embed(sample_text))
    late_embeddings = [list(vectors) for vectors in settings.late_interaction_embedding_model.embed(sample_text)]
    CreateCollection(
        collection_name=collection,
        dense_embeddings=dense_embeddings,
        late_embeddings=late_embeddings,
    ).build_collection()
