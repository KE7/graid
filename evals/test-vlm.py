from guidance import models, select


def main():
    gpt = models.Transformers("gpt2")

    resp = gpt + f"Do you want a joke or a poem? A " + select(["joke", "poem"])
    print(resp)


if __name__ == "__main__":
    main()
