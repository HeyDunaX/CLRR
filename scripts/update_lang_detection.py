with open("src/amis_rewire/train.py", "r", encoding="utf-8") as f:
    c = f.read()

old_mbart = """def detect_mbart_languages(data_dir: Path | str, cli_src: str | None, cli_tgt: str | None, cli_tok: str | None) -> tuple[str, str, str]:
    path_str = str(data_dir).lower()
    is_spanish = "ashaninka" in path_str or "spanish" in path_str
    tgt_lang = cli_tgt or ("es_XX" if is_spanish else "zh_CN")
    src_lang = cli_src or ("es_XX" if is_spanish else "tl_XX")
    bleu_tok = cli_tok or ("13a" if is_spanish else "zh")
    return src_lang, tgt_lang, bleu_tok"""

new_mbart = """def detect_mbart_languages(data_dir: Path | str, cli_src: str | None, cli_tgt: str | None, cli_tok: str | None) -> tuple[str, str, str]:
    path_str = str(data_dir).lower()
    is_spanish = "ashaninka" in path_str or "spanish" in path_str
    is_turkish = "turkish" in path_str or "tr_en" in path_str or "opus100" in path_str
    if is_turkish:
        tgt_lang = cli_tgt or "en_XX"
        src_lang = cli_src or "tr_TR"
        bleu_tok = cli_tok or "13a"
    elif is_spanish:
        tgt_lang = cli_tgt or "es_XX"
        src_lang = cli_src or "es_XX"
        bleu_tok = cli_tok or "13a"
    else:
        tgt_lang = cli_tgt or "zh_CN"
        src_lang = cli_src or "tl_XX"
        bleu_tok = cli_tok or "zh"
    return src_lang, tgt_lang, bleu_tok"""

c = c.replace(old_mbart.replace("\r\n", "\n"), new_mbart)
c = c.replace(old_mbart, new_mbart)
with open("src/amis_rewire/train.py", "w", encoding="utf-8") as f:
    f.write(c)
print("Updated src/amis_rewire/train.py")

with open("src/nllb_suite/train_nllb.py", "r", encoding="utf-8") as f:
    c_nllb = f.read()

old_nllb = """def detect_nllb_languages(data_dir: Path | str, cli_src: str | None, cli_tgt: str | None, cli_tok: str | None) -> tuple[str, str, str]:
    path_str = str(data_dir).lower()
    is_spanish = "ashaninka" in path_str or "spanish" in path_str
    tgt_lang = cli_tgt or ("spa_Latn" if is_spanish else "zho_Hant")
    src_lang = cli_src or ("spa_Latn" if is_spanish else "zho_Hant")
    bleu_tok = cli_tok or ("13a" if is_spanish else "zh")
    return src_lang, tgt_lang, bleu_tok"""

new_nllb = """def detect_nllb_languages(data_dir: Path | str, cli_src: str | None, cli_tgt: str | None, cli_tok: str | None) -> tuple[str, str, str]:
    path_str = str(data_dir).lower()
    is_spanish = "ashaninka" in path_str or "spanish" in path_str
    is_turkish = "turkish" in path_str or "tr_en" in path_str or "opus100" in path_str
    if is_turkish:
        tgt_lang = cli_tgt or "eng_Latn"
        src_lang = cli_src or "tur_Latn"
        bleu_tok = cli_tok or "13a"
    elif is_spanish:
        tgt_lang = cli_tgt or "spa_Latn"
        src_lang = cli_src or "spa_Latn"
        bleu_tok = cli_tok or "13a"
    else:
        tgt_lang = cli_tgt or "zho_Hant"
        src_lang = cli_src or "zho_Hant"
        bleu_tok = cli_tok or "zh"
    return src_lang, tgt_lang, bleu_tok"""

c_nllb = c_nllb.replace(old_nllb.replace("\r\n", "\n"), new_nllb)
c_nllb = c_nllb.replace(old_nllb, new_nllb)
with open("src/nllb_suite/train_nllb.py", "w", encoding="utf-8") as f:
    f.write(c_nllb)
print("Updated src/nllb_suite/train_nllb.py")
