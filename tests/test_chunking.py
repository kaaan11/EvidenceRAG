from app.rag.chunking import chunk_documents
from app.rag.config import InputLimitError
import pytest
import json
import hashlib
from pathlib import Path


def test_chunking_preserves_metadata_and_overlap():
    pages = [
        {
            "text": "one two three four five six seven eight nine ten",
            "metadata": {
                "filename": "notes.txt",
                "document_id": "doc-1",
                "page": None,
                "source": "notes.txt",
            },
        }
    ]

    chunks = chunk_documents(pages, chunk_size=20, chunk_overlap=5)

    assert len(chunks) > 1
    assert chunks[0]["metadata"]["filename"] == "notes.txt"
    assert chunks[0]["metadata"]["chunk_id"] == "doc-1-0"
    assert chunks[0]["text"].split()[-1] in chunks[1]["text"].split()


@pytest.mark.parametrize("count", [1, 2, 3])
def test_chunk_count_boundary(count):
    pages = [{"text": "x", "metadata": {"document_id": str(i)}} for i in range(count)]
    if count > 2:
        with pytest.raises(InputLimitError, match="chunks"):
            chunk_documents(pages, max_chunks=2)
    else:
        assert len(chunk_documents(pages, max_chunks=2)) == count


def test_chunk_generation_stops_without_consuming_unbounded_pages():
    consumed = []
    def pages():
        for i in range(100):
            consumed.append(i)
            yield {"text": "x", "metadata": {"document_id": str(i)}}
    with pytest.raises(InputLimitError):
        chunk_documents(pages(), max_chunks=2)
    assert consumed == [0, 1, 2]


def test_one_large_page_does_not_build_all_chunks_before_rejecting(monkeypatch):
    import app.rag.chunking as module
    original = module._split_text
    emitted = []
    def measured(*args):
        for part in original(*args):
            emitted.append(part)
            yield part
    monkeypatch.setattr(module, "_split_text", measured)
    with pytest.raises(InputLimitError):
        chunk_documents([{"text": "x" * 1000, "metadata": {"document_id": "d"}}],
                        chunk_size=5, chunk_overlap=0, max_chunks=2)
    assert len(emitted) == 3


def structured(text,size=800,overlap=100,**kwargs):
    return chunk_documents([{'text':text,'metadata':{'filename':'fixture.md','document_id':'d','page':None}}],
        chunk_size=size,chunk_overlap=overlap,strategy='structured_v2',ingestion_version=2,**kwargs)


def test_legacy_output_snapshot_and_explicit_default_retained():
    fixture=json.loads((Path(__file__).parent/'fixtures/chunking/legacy-baseline.json').read_text(encoding='utf-8'))
    pages=[{'text':fixture['text'],'metadata':{'filename':'structure.md','document_id':'fixture','page':None}}]
    assert chunk_documents(pages,fixture['chunk_size'],fixture['chunk_overlap'])==fixture['chunks']
    from app.rag.config import Settings
    assert Settings().chunker_strategy=='legacy_character'
    with pytest.raises(ValueError):Settings(chunker_strategy='structured_v2')


def test_structure_lines_headings_and_exact_source_round_trip():
    text=(Path(__file__).parent/'fixtures/chunking/structure.md').read_text(encoding='utf-8')
    chunks=structured(text)
    assert '| Yıl | Tutar |\n| 2025 | 5.800 |' in chunks[0]['text']
    assert '- İlk işlem\n- İkinci işlem' in chunks[0]['text']
    assert chunks[-1]['metadata']['heading_path']==['Belediye','İzin']
    for chunk in chunks:
        m=chunk['metadata']
        assert text[m['char_start']:m['char_end']]==chunk['text']
        assert m['source_sha256']==hashlib.sha256(text.encode()).hexdigest()
        assert m['offset_unit']=='unicode_character' and m['normalization_version']==2


@pytest.mark.parametrize('text',['',' \t\n','# İzin\n\nŞükriye için çağrı.','alpha bravo charlie delta echo foxtrot '*30,'x'*1000])
@pytest.mark.parametrize('size,overlap',[(1,0),(10,0),(10,9),(40,7)])
def test_structured_unicode_empty_overlap_and_long_token_bounds(text,size,overlap):
    chunks=structured(text,size,overlap,max_chunks=2000)
    assert len(chunks)<=2000
    starts=[]
    for chunk in chunks:
        m=chunk['metadata'];starts.append(m['char_start'])
        assert 0<len(chunk['text'])<=size
        assert text[m['char_start']:m['char_end']]==chunk['text']
    assert starts==sorted(set(starts))


def test_overlap_never_starts_inside_standard_words_and_hard_split_is_explicit():
    text='alpha bravo charlie delta echo foxtrot golf hotel '*5
    chunks=structured(text,23,8)
    for chunk in chunks:
        m=chunk['metadata'];at=m['char_start']+len(chunk['text'])-len(chunk['text'].lstrip())
        assert at==0 or text[at-1].isspace()
        assert not m['hard_split']
        assert not chunk['text'].startswith(('lie ','cho '))
    hard=structured('x'*31,10,3)
    assert any(c['metadata']['hard_split'] for c in hard)
    assert all(len(c['text'])<=10 for c in hard)


def test_source_representation_crlf_and_page_offsets_are_local():
    pages=[{'text':'alpha\r\nbeta','metadata':{'filename':'a.pdf','document_id':'doc','page':1}},
           {'text':'gamma delta','metadata':{'filename':'a.pdf','document_id':'doc','page':2}}]
    chunks=chunk_documents(pages,strategy='structured_v2',ingestion_version=2,chunk_size=20,chunk_overlap=0)
    assert [c['metadata']['page'] for c in chunks]==[1,2]
    for c,page in zip(chunks,pages):
        source=page['text'].replace('\r\n','\n');m=c['metadata']
        assert source[m['char_start']:m['char_end']]==c['text']
        assert m['filename']=='a.pdf' and m['document_id']=='doc'
        assert m['source_representation']=='extracted_page_lf_v1'


def test_native_pdf_pages_keep_separate_spans(tmp_path):
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject,DictionaryObject,NameObject
    from app.rag.document_loader import load_document
    writer=PdfWriter()
    for value in ['first page','second page']:
        page=writer.add_blank_page(width=100,height=100)
        font=DictionaryObject({NameObject('/Type'):NameObject('/Font'),NameObject('/Subtype'):NameObject('/Type1'),NameObject('/BaseFont'):NameObject('/Helvetica')})
        page[NameObject('/Resources')]=DictionaryObject({NameObject('/Font'):DictionaryObject({NameObject('/F1'):font})})
        stream=DecodedStreamObject();stream.set_data(f'BT /F1 12 Tf 10 10 Td ({value}) Tj ET'.encode())
        page[NameObject('/Contents')]=writer._add_object(stream)
    path=tmp_path/'pages.pdf';writer.write(path)
    pages=load_document(path);chunks=chunk_documents(pages,strategy='structured_v2',ingestion_version=2)
    assert [c['metadata']['page'] for c in chunks]==[1,2]
    for c,page in zip(chunks,pages):
        m=c['metadata'];assert page['text'][m['char_start']:m['char_end']]==c['text']


def test_t08_source_integrity_and_fixed_denominator_for_structured_coverage():
    from experiments.eval.scoring import sha,retrieval_row,retrieval_metrics
    text='alpha bravo charlie delta echo';corpus={'fixture.md':{'text':text,'sha256':sha(text.encode())}}
    chunks=[{**c,**c['metadata'],'score':1.0} for c in structured(text,12,0)]
    gold={'document':'fixture.md','sha256':sha(text.encode()),'start':0,'end':len(text),'text':text}
    q={'id':'q','question':'fact?','category':'answerable'}
    row=retrieval_row(q,[[gold]],corpus,chunks,chunks,.8,100)
    assert not row['oracle_invalid'] and not row['chunk_coverage_failure'] and row['rank'] is not None
    rows=[{**row,'id':str(i)} for i in range(9)]
    lost=retrieval_row({**q,'id':'lost'},[[gold]],corpus,[],[],.8,100);rows.append(lost)
    assert lost['chunk_coverage_failure'] and not lost['oracle_invalid']
    assert retrieval_metrics(rows)['answerable']['count']==10
    assert retrieval_metrics(rows)['answerable']['hit_at_k']==.9
