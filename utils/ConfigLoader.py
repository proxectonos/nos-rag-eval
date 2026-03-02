from dataclasses import dataclass
from typing import Optional, List
from itertools import product
from types import SimpleNamespace
import yaml

# ---------------------------------------------------------------------------
# Original dataclasses — kept for backward compatibility with any code
# that still uses ConfigLoader.load() with the old per-experiment YAMLs.
# ---------------------------------------------------------------------------

@dataclass
class GeneralConfig:
    hf_cache_dir: str
    dataset_name : Optional[str] = None

@dataclass
class DatabaseConfig:
    elastic_index: str
    elastic_config_file: str
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None

@dataclass
class RetrieverConfig:
    retrieval_strategy: str
    num_docs_retrieval: int
    num_docs_reranker: int
    embedding_model: Optional[str] = None

@dataclass
class RerankerConfig:
    use_reranking: bool
    reranker_model: Optional[str] = None

@dataclass
class ElasticConfig:
    username: str
    password: str
    url: str
    endpoint: str

@dataclass
class Config:
    general_config: GeneralConfig
    database: DatabaseConfig
    retriever: RetrieverConfig
    reranker: RerankerConfig


class ConfigLoader:
    @staticmethod
    def load(config_path) -> Config:
        with open(config_path, 'r') as file:
            config_dict = yaml.safe_load(file)

        return Config(
            general_config=GeneralConfig(**config_dict['general_config']),
            database=DatabaseConfig(**config_dict['database']),
            retriever=RetrieverConfig(**config_dict['retriever']),
            reranker=RerankerConfig(**config_dict['reranker'])
        )

    @staticmethod
    def load_elastic(config_path) -> ElasticConfig:
        with open(config_path, 'r') as file:
            config_dict = yaml.safe_load(file)

        return ElasticConfig(
            username=config_dict['username'],
            password=config_dict['password'],
            url=config_dict['elastic_url'],
            endpoint=config_dict['api_endpoint']
        )


# ---------------------------------------------------------------------------
# New dataclasses for the unified experiments YAML
# ---------------------------------------------------------------------------

@dataclass
class _GeneralConfig:
    hf_cache_dir: str
    elastic_config_file: str
    dataset_name: str

@dataclass
class _RetrieverDefaults:
    retrieval_strategy: str
    num_docs_retrieval: int
    num_docs_reranker: int

@dataclass
class _EmbedderConfig:
    name: str
    elastic_index: str
    embedding_model: Optional[str] = None
    retrieval_strategy: Optional[str] = None  # overrides retriever_defaults when set (e.g. BM25)

@dataclass
class _RerankerConfig:
    name: str
    reranker_model: Optional[str] = None

    @property
    def use_reranking(self) -> bool:
        return self.name != "none"


@dataclass
class ExperimentConfig:
    """
    A fully resolved single experiment — equivalent to one of the old per-experiment YAMLs.
    Exposes .database, .retriever, .reranker, and .general_config as SimpleNamespace
    objects so that RAG.__init__ works without any changes.
    """
    name: str
    hf_cache_dir: str
    elastic_config_file: str
    dataset_name: str
    embedding_model: Optional[str]
    elastic_index: str
    retrieval_strategy: str
    num_docs_reranker: int
    num_docs_retrieval: int
    use_reranking: bool
    reranker_model: Optional[str]

    @property
    def general_config(self) -> SimpleNamespace:
        return SimpleNamespace(
            hf_cache_dir=self.hf_cache_dir,
            elastic_config_file=self.elastic_config_file,
            dataset_name=self.dataset_name
        )

    @property
    def database(self) -> SimpleNamespace:
        return SimpleNamespace(
            elastic_index=self.elastic_index,
            elastic_config_file=self.elastic_config_file
        )

    @property
    def retriever(self) -> SimpleNamespace:
        return SimpleNamespace(
            embedding_model=self.embedding_model,
            retrieval_strategy=self.retrieval_strategy,
            num_docs_retrieval=self.num_docs_retrieval,
            num_docs_reranker=self.num_docs_reranker
        )

    @property
    def reranker(self) -> SimpleNamespace:
        return SimpleNamespace(
            use_reranking=self.use_reranking,
            reranker_model=self.reranker_model
        )


class ExperimentsLoader:
    @staticmethod
    def load(config_path: str) -> List[ExperimentConfig]:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)

        general = _GeneralConfig(**cfg['general_config'])
        defaults = _RetrieverDefaults(**cfg['retriever_defaults'])

        embedders = {e['name']: _EmbedderConfig(**e) for e in cfg['embedders']}
        rerankers = {r['name']: _RerankerConfig(**r) for r in cfg['rerankers']}

        # Use explicit experiment list if defined, otherwise run all combinations
        if 'experiments' in cfg:
            pairs = [(exp['embedder'], exp['reranker']) for exp in cfg['experiments']]
        else:
            pairs = list(product(embedders.keys(), rerankers.keys()))

        experiments = []
        for emb_name, rer_name in pairs:
            emb = embedders[emb_name]
            rer = rerankers[rer_name]
            experiments.append(ExperimentConfig(
                name=f"{emb_name}_{rer_name}",
                hf_cache_dir=general.hf_cache_dir,
                elastic_config_file=general.elastic_config_file,
                dataset_name=general.dataset_name,
                embedding_model=emb.embedding_model,
                elastic_index=emb.elastic_index,
                retrieval_strategy=emb.retrieval_strategy or defaults.retrieval_strategy,
                num_docs_retrieval=defaults.num_docs_retrieval,
                num_docs_reranker=defaults.num_docs_reranker,
                use_reranking=rer.use_reranking,
                reranker_model=rer.reranker_model,
            ))

        return experiments


if __name__ == "__main__":
    experiments = ExperimentsLoader.load("experiments.yaml")
    for exp in experiments:
        print(f"\nExperiment: {exp.name}")
        print("  general_config:", exp.general_config)
        print("  database:      ", exp.database)
        print("  retriever:     ", exp.retriever)
        print("  reranker:      ", exp.reranker)
