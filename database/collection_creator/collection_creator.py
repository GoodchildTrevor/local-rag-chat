from qdrant_client.models import models

from config.settings import get_settings

settings = get_settings()


class CreateCollection:
    """
    Utility class for creating or recreating a Qdrant collection
    with dense, sparse, and optionally late-interaction embeddings.
    :param collection_name: Name of the collection to create or recreate.
    :type collection_name: str
    :param dense_embeddings: List of dense embedding vectors.
    :type dense_embeddings: list[list[float]]
    :param late_embeddings: Optional 3D list of late interaction vectors for multi-vector configuration.
    :type late_embeddings: list[list[list[float]], optional
    :param late: Flag to include late interaction vector configuration.
    :type late: bool, optional
    :param recreation: Whether to recreate the collection (drops and creates anew).
    :type recreation: bool, optional
    """

    def __init__(
        self,
        collection_name: str,
        dense_embeddings: list[list[float]],
        late_embeddings: list[list[list[float]]] = None,
        late: bool = False,
        recreation: bool = False
    ):
        self.client = settings.client
        self.collection_name = collection_name
        self.dense_embeddings = dense_embeddings
        self.late_embeddings = late_embeddings
        self.late = late
        self.recreation = recreation

    def creator(self, configs: dict) -> None:
        """
        Creates a Qdrant collection with the given configuration.
        :param configs: Dictionary containing vector and sparse vector configurations.
        """
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=configs["vectors_config"],
            sparse_vectors_config=configs["sparse_vectors_config"],
        )

    def recreator(self, configs: dict) -> None:
        """
        Recreates (deletes and creates) a Qdrant collection with the given configuration.
        :param configs: Dictionary containing vector and sparse vector configurations.
        """
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=configs["vectors_config"],
            sparse_vectors_config=configs["sparse_vectors_config"],
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
            "sparse_vectors_config": sparse_config,
        }

        if self.recreation:
            self.recreator(configs)
        else:
            self.creator(configs)


if __name__ == "__main__":
    sample_text = "Hello World"
    dense_embeddings = list(settings.dense_embedding_model.embed(sample_text))
    late_embeddings = [list(vectors) for vectors in settings.late_interaction_embedding_model.embed(sample_text)]
    CreateCollection(
        collection_name="documents",
        dense_embeddings=dense_embeddings,
        late_embeddings=late_embeddings,
        late = True,
        recreation=True,
    ).build_collection()