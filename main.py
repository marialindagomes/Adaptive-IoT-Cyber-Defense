"""
Main Pipeline

Explainable AI-Based Intelligent System
for Real-Time Cyber Threat Detection
and Adaptive Response

"""


from pathlib import Path


CONFIG_PATH = Path("config.yaml")


def load_config():

    if not CONFIG_PATH.exists():

        raise FileNotFoundError(
            "config.yaml file not found"
        )

    config = {}
    section = None

    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        for raw_line in file:
            line = raw_line.rstrip()

            if not line.strip() or line.lstrip().startswith("#"):
                continue

            if not line.startswith((" ", "\t")) and line.endswith(":"):
                section = line[:-1].strip()
                config[section] = {}
                continue

            if section and ":" in line:
                key, value = line.split(":", 1)
                config[section][key.strip()] = value.strip().strip("\"'")

    return config


def main():

    print("="*60)

    print(
        "Explainable AI Cyber Threat Detection System"
    )

    print("="*60)

    config = load_config()

    print("\nProject:")
    print(
        config["project"]["name"]
    )

    print("\nPrimary Dataset:")
    print(
        config["dataset"]["primary_dataset"]
    )

    print("\nSecondary Dataset:")
    print(
        config["dataset"]["secondary_dataset"]
    )

    print("\nPipeline Status:")
    print(
        "Configuration loaded successfully"
    )

    print("\nNext step:")
    print(
        "Run dataset inspection pipeline"
    )


if __name__ == "__main__":

    main()
