import pytest
from chains.task_chain import TaskChain

@pytest.fixture
def sample_syllabus():
    return "This is a sample syllabus for a class on Python programming. The main topics covered are algorithms, data structures, and object-oriented programming."

@pytest.fixture
def task_chain(sample_syllabus):
    return TaskChain(syllabus=sample_syllabus)

def test_task_chain_init(task_chain, sample_syllabus):
    assert task_chain.syllabus == sample_syllabus
    assert task_chain.llm is not None
    assert task_chain.prompt is not None
    assert task_chain.chain is not None

def test_task_chain_run(task_chain):
    question = "What topics does the class cover?"
    response = task_chain.run(question)

    assert "algorithms" in response
    assert "data structures" in response
    assert "object-oriented programming" in response
