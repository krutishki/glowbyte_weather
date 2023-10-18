import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run an energy consumption prediction model for Glowbyte by Krutishki team')
    parser.add_argument('-p', type=str, help='Path to input CSV file')
    args = parser.parse_args()

    file_path = args.p
    print(file_path)
