from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
CONCEPTS_PATH = DATA_DIR / "concepts.json"
DOCUMENTS_PATH = DATA_DIR / "documents.json"
EXTRACTED_DIR = DATA_DIR / "extracted"
PARSED_DIR = DATA_DIR / "parsed"

ONTOLOGY_DIR = PROJECT_ROOT / "ontology"
GENERATED_DIR = ONTOLOGY_DIR / "generated"
GENERATED_ONTOLOGY_PATH = GENERATED_DIR / "legal-documents.rdf"
ONTOLOGY_SCHEMA_PATH = ONTOLOGY_DIR / "legal-ontology.rdf"

ONTOLOGY_IRI = "https://example.org/legal-ontology"
DATA_ONTOLOGY_IRI = f"{ONTOLOGY_IRI}/data/legal-documents"
