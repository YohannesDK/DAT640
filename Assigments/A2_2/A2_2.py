import abc
from collections import Counter
from collections import UserDict as DictClass
from collections import defaultdict
from typing import Dict, List
import math

from attr import field

CollectionType = Dict[str, Dict[str, List[str]]]


class DocumentCollection(DictClass):
    """Document dictionary class with helper functions."""

    def total_field_length(self, field: str) -> int:
        """Total number of terms in a field for all documents."""
        return sum(len(fields[field]) for fields in self.values())

    def avg_field_length(self, field: str) -> float:
        """Average number of terms in a field across all documents."""
        return self.total_field_length(field) / len(self)

    def get_field_documents(self, field: str) -> Dict[str, List[str]]:
        """Dictionary of documents for a single field."""
        return {
            doc_id: doc[field] for (doc_id, doc) in self.items() if field in doc
        }


class Scorer(abc.ABC):
    def __init__(
        self,
        collection: DocumentCollection,
        index: CollectionType,
        field: str = None,
        fields: List[str] = None,
    ):
        """Interface for the scorer class.

        Args:
            collection: Collection of documents. Needed to calculate document
                statistical information.
            index: Index to use for calculating scores.
            field (optional): Single field to use in scoring.. Defaults to None.
            fields (optional): List of fields to use in scoring. Defaults to
                None.

        Raises:
            ValueError: Either field or fields need to be specified.
        """
        self.collection = collection
        self.index = index

        if not (field or fields):
            raise ValueError("Either field or fields have to be defined.")

        self.field = field
        self.fields = fields

        # Score accumulator for the query that is currently being scored.
        self.scores = None

    def score_collection(self, query_terms: List[str]):
        """Scores all documents in the collection using term-at-a-time query
        processing.

        Params:
            query_term: Sequence (list) of query terms.

        Returns:
            Dict with doc_ids as keys and retrieval scores as values.
            (It may be assumed that documents that are not present in this dict
            have a retrival score of 0.)
        """
        self.scores = defaultdict(float)  # Reset scores.
        query_term_freqs = Counter(query_terms)

        for term, query_freq in query_term_freqs.items():
            self.score_term(term, query_freq)

        return self.scores

    @abc.abstractmethod
    def score_term(self, term: str, query_freq: int):
        """Scores one query term and updates the accumulated document retrieval
        scores (`self.scores`).

        Params:
            term: Query term
            query_freq: Frequency (count) of the term in the query.
        """
        raise NotImplementedError


class SimpleScorer(Scorer):
    def score_term(self, term: str, query_freq: int) -> None:
        for doc_id, doc in self.collection.items():
            if term in doc[self.field]:
                self.scores[doc_id] += doc[self.field].count(term) * query_freq
    

class ScorerBM25(Scorer):
    def __init__(
        self,
        collection: DocumentCollection,
        index: CollectionType,
        field: str = "body",
        b: float = 0.75,
        k1: float = 1.2,
    ) -> None:
        super(ScorerBM25, self).__init__(collection, index, field)
        self.b = b
        self.k1 = k1

    def score_term(self, term: str, query_freq: int) -> None:
        for doc_id, doc in self.collection.items():
            if term in doc[self.field]:
                idf = math.log(len(self.collection) / len(self.index[self.field][term]))
                c_t_d = doc[self.field].count(term)
                numerator = c_t_d * (1 + self.k1)
                denominator = c_t_d + self.k1 * (1 - self.b + self.b * len(doc[self.field]) / self.collection.avg_field_length(self.field))
                self.scores[doc_id] += (numerator / denominator) * idf
        


class ScorerLM(Scorer):
    def __init__(
        self,
        collection: DocumentCollection,
        index: CollectionType,
        field: str = "body",
        smoothing_param: float = 0.1,
    ):
        super(ScorerLM, self).__init__(collection, index, field)
        self.smoothing_param = smoothing_param

    def score_term(self, term: str, query_freq: int) -> None:
        # TODO - fix later
        c_t_q = query_freq
        lambda_ = self.smoothing_param
        P_t_C = sum([doc[self.field].count(term) for doc in self.collection.values()]) / self.collection.total_field_length(self.field)

        for doc_id, doc in self.collection.items():
            if term in doc[self.field]:
                c_t_d = doc[self.field].count(term)
                self.scores[doc_id] += c_t_q * math.log((1 - lambda_) * (c_t_d / len(doc[self.field])) + (lambda_ * P_t_C))

class ScorerBM25F(Scorer):
    def __init__(
        self,
        collection: DocumentCollection,
        index: CollectionType,
        fields: List[str] = ["title", "body"],
        field_weights: List[float] = [0.2, 0.8],
        bi: List[float] = [0.75, 0.75],
        k1: float = 1.2,
    ) -> None:
        super(ScorerBM25F, self).__init__(collection, index, fields=fields)
        self.field_weights = field_weights
        self.bi = bi
        self.k1 = k1

    def score_term(self, term: str, query_freq: int) -> None:
        for doc_id, doc in self.collection.items():
            c_t_d = 0
            idf = 0
            for field_index, field in enumerate(self.fields):
                B_i = (1 - self.bi[field_index] + self.bi[field_index] * len(doc[field]) / self.collection.avg_field_length(field))
                c_t_d += self.field_weights[field_index] * (doc[field].count(term) / B_i)
                if field == "body":
                    idf = math.log(len(self.collection) / len(self.index[field][term]))
            self.scores[doc_id] += (c_t_d / (self.k1 + c_t_d)) * idf
            
class ScorerMLM(Scorer):
    def __init__(
        self,
        collection: DocumentCollection,
        index: CollectionType,
        fields: List[str] = ["title", "body"],
        field_weights: List[float] = [0.2, 0.8],
        smoothing_param: float = 0.1,
    ):
        super(ScorerMLM, self).__init__(collection, index, fields=fields)
        self.field_weights = field_weights
        self.smoothing_param = smoothing_param

    def score_term(self, term: str, query_freq: float) -> None:
        for doc_id, doc in self.collection.items():
            c_t_d = 0
            P_t_C = 0
            lambda_ = self.smoothing_param
            for field_index, field in enumerate(self.fields):
                c_t_d += self.field_weights[field_index] * (doc[field].count(term) / len(doc[field]))
                P_t_C += self.field_weights[field_index] * (sum([doc[field].count(term) for doc in self.collection.values()]) / self.collection.total_field_length(field))
            self.scores[doc_id] += math.log((1 - lambda_) * c_t_d + lambda_ * P_t_C)
        
            

