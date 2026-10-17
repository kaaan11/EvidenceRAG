"""All automated lanes are model-offline; native means real framework/store operations."""
import os
import pytest

os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'


def pytest_configure(config):
    for name,description in [('native','Real FAISS/framework/files/OS-lock integration, fake models'),
        ('model_free','Model-free algorithm/adapter tests'),('real_model','Explicit cache-only real-model integration')]:
        config.addinivalue_line('markers',name+': '+description)


def pytest_collection_modifyitems(items):
    native_files={'test_api.py','test_pipeline.py','test_vector_store.py','test_storage_lease.py',
        'test_retrieval.py','test_runtime.py','test_evaluation.py','test_end_to_end.py'}
    for item in items:
        if item.get_closest_marker('real_model'):
            if os.getenv('EVIDENCERAG_REAL_MODEL_TESTS')!='1':
                item.add_marker(pytest.mark.skip(reason='Real-model lane needs explicit opt-in and an existing immutable cache'))
        else:
            item.add_marker(pytest.mark.native if item.path.name in native_files or 'native_pdf' in item.name else pytest.mark.model_free)
