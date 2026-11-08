import io
import zipfile
from unittest.mock import patch

import pytest

from app.uploads import extract_document, make_chunks, new_collection_name


def test_text_and_markdown_chunks():
    chunks = make_chunks([('notes.txt', b'Our vacation policy is 15 days.'),
                          ('readme.md', b'# Company info\nHours are flexible.')])
    assert len(chunks) == 2
    assert {c['source'] for c in chunks} == {'notes.txt', 'readme.md'}


def test_docx_without_new_library():
    xml = (b'<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
           b'<w:body><w:p><w:r><w:t>Benefits policy</w:t></w:r></w:p></w:body></w:document>')
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        archive.writestr('word/document.xml', xml)
    assert extract_document('benefits.docx', buffer.getvalue()) == [('benefits.docx', 'Benefits policy')]


def test_invalid_files_and_empty_text():
    with pytest.raises(ValueError):
        extract_document('script.sh', b'echo hi')
    with pytest.raises(ValueError):
        make_chunks([('empty.txt', b'   ')])


def test_collections_are_unique():
    assert new_collection_name() != new_collection_name()


def test_uploaded_questions_skip_sample_retrieval_and_redis():
    from app import service
    docs = [{'source': 'my_notes.txt', 'text': 'A new return period is 45 days.'}]
    with patch('app.uploads.search_uploads', return_value=docs), \
         patch('app.service.generate', return_value=('45 days', 'ollama:gemma3:1b')), \
         patch('app.service.retrieve') as sample_retrieve, \
         patch('app.service.redis.Redis.from_url') as cache:
        answer = service.answer_uploaded_question('What is the return period?', new_collection_name())
        assert answer.answer == '45 days'
        assert answer.citations[0].source == 'my_notes.txt'
        assert answer.cached is False
        sample_retrieve.assert_not_called()
        cache.assert_not_called()
