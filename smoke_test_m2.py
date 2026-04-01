import os
import pickle
import traceback


def safe_text(value):
    text = str(value)
    return text.encode("gbk", errors="replace").decode("gbk", errors="replace")


def print_header(title):
    print("\n" + "=" * 80)
    print(safe_text(title))
    print("=" * 80)


def run_step(name, fn):
    print(safe_text(f"[STEP] {name}"))
    try:
        result = fn()
        print(safe_text(f"[OK]   {name}"))
        return True, result
    except Exception as exc:
        print(safe_text(f"[FAIL] {name}"))
        print(safe_text(f"Exception: {exc.__class__.__name__}: {exc}"))
        print("Traceback:")
        traceback.print_exc()
        return False, None


def main():
    print_header("M2 Transformer Smoke Test")

    imported = {}

    def step_imports():
        from models.transformer import (
            Transformer,
            MemoryAugmentedEncoder,
            MeshedDecoder,
            ScaledDotProductAttentionMemory,
        )
        imported["Transformer"] = Transformer
        imported["MemoryAugmentedEncoder"] = MemoryAugmentedEncoder
        imported["MeshedDecoder"] = MeshedDecoder
        imported["ScaledDotProductAttentionMemory"] = ScaledDotProductAttentionMemory
        print("Imported symbols:")
        for key in imported:
            print(safe_text(f"  - {key}: {imported[key]}"))

    ok_imports, _ = run_step("Import models.transformer related modules", step_imports)

    vocab_state = {}

    def step_vocab():
        vocab_path = "vocab.pkl"
        print(safe_text(f"Checking file exists: {os.path.abspath(vocab_path)}"))
        print(safe_text(f"Exists: {os.path.isfile(vocab_path)}"))
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)

        vocab_state["obj"] = vocab
        print(safe_text(f"Loaded vocab type: {type(vocab)}"))
        print(safe_text(f"Has stoi: {hasattr(vocab, 'stoi')}"))
        print(safe_text(f"Has itos: {hasattr(vocab, 'itos')}"))

        if hasattr(vocab, "stoi"):
            for token in ["<bos>", "<eos>", "<pad>"]:
                print(safe_text(f"Token {token!r} in stoi: {token in vocab.stoi}"))
                if token in vocab.stoi:
                    print(safe_text(f"  stoi[{token!r}] = {vocab.stoi[token]}"))

        if hasattr(vocab, "itos"):
            print(safe_text(f"itos length: {len(vocab.itos)}"))

    ok_vocab, _ = run_step("Load vocab.pkl and inspect stoi/itos/special tokens", step_vocab)

    def step_encoder():
        encoder = imported["MemoryAugmentedEncoder"](
            3,
            0,
            attention_module=imported["ScaledDotProductAttentionMemory"],
            attention_module_kwargs={"m": 40},
        )
        print(safe_text(f"Encoder instance type: {type(encoder)}"))
        return encoder

    ok_encoder, encoder = run_step("Instantiate MemoryAugmentedEncoder", step_encoder)

    def step_decoder():
        vocab = vocab_state["obj"]
        decoder = imported["MeshedDecoder"](
            len(vocab),
            54,
            3,
            vocab.stoi["<pad>"],
        )
        print(safe_text(f"Decoder instance type: {type(decoder)}"))
        return decoder

    ok_decoder, decoder = run_step("Instantiate MeshedDecoder", step_decoder)

    def step_transformer():
        vocab = vocab_state["obj"]
        model = imported["Transformer"](
            vocab.stoi["<bos>"],
            encoder,
            decoder,
        )
        print(safe_text(f"Transformer instance type: {type(model)}"))
        return model

    ok_transformer, _ = run_step("Instantiate Transformer", step_transformer)

    print_header("Summary")
    print(safe_text(f"Import models.transformer modules: {ok_imports}"))
    print(safe_text(f"Load vocab.pkl: {ok_vocab}"))
    print(safe_text(f"Instantiate MemoryAugmentedEncoder: {ok_encoder}"))
    print(safe_text(f"Instantiate MeshedDecoder: {ok_decoder}"))
    print(safe_text(f"Instantiate Transformer: {ok_transformer}"))


if __name__ == "__main__":
    main()
