"""
DNA Data Hiding Scheme Implementation
Based on: "A New Data Hiding Scheme Based on DNA Sequence"
by Guo, Chang, and Wang (2012)

This implementation uses complementary rules to hide 2 bits per character
in repeated positions of a DNA sequence.
"""

import os
import sys


class DNASteganography:
    def __init__(self, complementary_rule=None):
        """
        Initialize with a complementary rule.
        Default rule: (AT)(CA)(GC)(TG)
        C(A)=T, C(C)=A, C(G)=C, C(T)=G
        """
        if complementary_rule is None:
            # Default complementary rule from the paper
            self.complement = {'A': 'T', 'C': 'A', 'G': 'C', 'T': 'G'}
        else:
            self.complement = complementary_rule

    def C(self, x):
        """Apply complementary rule once"""
        return self.complement[x]

    def CC(self, x):
        """Apply complementary rule twice"""
        return self.C(self.C(x))

    def CCC(self, x):
        """Apply complementary rule three times"""
        return self.C(self.C(self.C(x)))

    def find_repeated_positions(self, sequence, version='updated'):
        """
        Find positions of repeated characters that can be used to hide Data.

        Basic version: Only the second consecutive repeated character
        Updated version: All consecutive repeated characters except the first

        Returns list of indices where Data can be hidden.
        """
        positions = []
        i = 0
        while i < len(sequence):
            # Find consecutive repeated characters
            j = i + 1
            while j < len(sequence) and sequence[j] == sequence[i]:
                j += 1

            # Number of consecutive repeated characters
            repeat_count = j - i

            if repeat_count >= 2:
                if version == 'basic':
                    # Basic version: only use the second repeated character
                    positions.append(i + 1)
                else:
                    # Updated version: use all repeated characters except the first
                    for k in range(i + 1, j):
                        positions.append(k)

            i = j if repeat_count > 1 else i + 1

        return positions

    def text_to_binary(self, text):
        """Convert text to binary string"""
        binary = ''.join(format(ord(c), '08b') for c in text)
        return binary

    def binary_to_text(self, binary):
        """Convert binary string to text"""
        # Pad to multiple of 8 if needed
        while len(binary) % 8 != 0:
            binary = '0' + binary

        text = ''
        for i in range(0, len(binary), 8):
            byte = binary[i:i + 8]
            text += chr(int(byte, 2))
        return text

    def hide_data(self, reference_sequence, secret_binary, version='updated'):

        if not all(c in '01' for c in secret_binary):
            raise ValueError("Secret message must be a binary string (only 0 and 1)")

        # Ensure even length
        if len(secret_binary) % 2 != 0:
            secret_binary += '0'

        # Find positions for hiding
        positions = self.find_repeated_positions(reference_sequence, version)

        # Check capacity
        max_bits = len(positions) * 2
        if len(secret_binary) > max_bits:
            raise ValueError(f"Secret message too long. Max capacity: {max_bits} bits, "
                             f"Message length: {len(secret_binary)} bits")

        # Convert sequence to list for modification
        fake_sequence = list(reference_sequence)

        # Split secret message into 2-bit segments
        segments = [secret_binary[i:i + 2] for i in range(0, len(secret_binary), 2)]

        # Hide Data
        for i, segment in enumerate(segments):
            pos = positions[i]
            original_char = reference_sequence[pos]

            if segment == '00':
                # Keep original
                new_char = original_char
            elif segment == '01':
                # C(x)
                new_char = self.C(original_char)
            elif segment == '10':
                # C(C(x))
                new_char = self.CC(original_char)
            elif segment == '11':
                # C(C(C(x)))
                new_char = self.CCC(original_char)

            fake_sequence[pos] = new_char

        return ''.join(fake_sequence), len(segments)

    def extract_data(self, fake_sequence, reference_sequence, num_segments, version='updated'):

        positions = self.find_repeated_positions(reference_sequence, version)

        extracted_bits = ''

        for i in range(num_segments):
            pos = positions[i]
            original_char = reference_sequence[pos]
            fake_char = fake_sequence[pos]

            if fake_char == original_char:
                extracted_bits += '00'
            elif fake_char == self.C(original_char):
                extracted_bits += '01'
            elif fake_char == self.CC(original_char):
                extracted_bits += '10'
            elif fake_char == self.CCC(original_char):
                extracted_bits += '11'

        return extracted_bits


def read_dna_sequence(filepath):
    """
    Read DNA sequence from file.
    Filters out comments (lines starting with #) and whitespace.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Filter out comments and whitespace
    sequence = ''
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            # Remove any whitespace in the sequence
            sequence += line.replace(' ', '').replace('\n', '')

    # Validate DNA sequence
    valid_chars = set('ACGT')
    sequence = sequence.upper()
    if not all(c in valid_chars for c in sequence):
        raise ValueError("DNA sequence contains invalid characters. Only A, C, G, T are allowed.")

    return sequence


def read_binary_message(filepath):
    """
    Read binary message from file.
    The file should contain only 0s and 1s.
    """
    with open(filepath, 'r') as f:
        content = f.read().strip()

    # Remove any whitespace
    binary = content.replace(' ', '').replace('\n', '').replace('\r', '')

    # Validate binary string
    if not all(c in '01' for c in binary):
        raise ValueError("Secret message file must contain only 0 and 1")

    return binary


def main():


    # Parse command line arguments
    if len(sys.argv) < 3:
        print("\nUsage: python dna_steganography.py <dna_file> <secret_file> [version]")
        print("\nArguments:")
        print("  dna_file    : Path to the original DNA sequence file")
        print("  secret_file : Path to the binary secret message file")
        print("  version     : 'basic' or 'updated' (default: 'updated')")
        print("\nExample:")
        print("  python dna_steganography.py original_DNA.txt secret.txt updated")
        return

    dna_file = sys.argv[1]
    secret_file = sys.argv[2]
    version = sys.argv[3] if len(sys.argv) > 3 else 'updated'

    if version not in ['basic', 'updated']:
        print("Error: version must be 'basic' or 'updated'")
        return

    # Check if files exist
    if not os.path.exists(dna_file):
        print(f"Error: DNA file '{dna_file}' not found")
        return

    if not os.path.exists(secret_file):
        print(f"Error: Secret file '{secret_file}' not found")
        return

    # Initialize the steganography system
    stego = DNASteganography()

    # Read input files
    print(f"\n[1] Reading input files...")
    try:
        reference_dna = read_dna_sequence(dna_file)
        secret_binary = read_binary_message(secret_file)
    except Exception as e:
        print(f"Error reading files: {e}")
        return

    print(f"Reference DNA length: {len(reference_dna)}")
    print(f"First 100 characters: {reference_dna[:100]}...")
    print(f"Secret message length: {len(secret_binary)} bits")
    print(f"First 80 bits: {secret_binary[:80]}...")

    # Check embedding capacity
    positions_basic = stego.find_repeated_positions(reference_dna, 'basic')
    positions_updated = stego.find_repeated_positions(reference_dna, 'updated')

    print(f"\n[2] Embedding capacity analysis:")
    print(f"Basic version: {len(positions_basic)} positions = {len(positions_basic) * 2} bits")
    print(f"Updated version: {len(positions_updated)} positions = {len(positions_updated) * 2} bits")

    # Check if secret message fits
    max_capacity = len(positions_updated) * 2 if version == 'updated' else len(positions_basic) * 2
    if len(secret_binary) > max_capacity:
        print(f"\nError: Secret message ({len(secret_binary)} bits) is too long!")
        print(f"Maximum capacity for {version} version: {max_capacity} bits")
        return

    # Hide Data
    print(f"\n[3] Hiding Data using {version} version...")
    try:
        fake_dna, num_segments = stego.hide_data(reference_dna, secret_binary, version)
    except Exception as e:
        print(f"Error hiding Data: {e}")
        return

    print(f"Number of 2-bit segments hidden: {num_segments}")
    print(f"Fake DNA length: {len(fake_dna)}")
    print(f"First 100 characters: {fake_dna[:100]}...")

    # Calculate modification rate
    modifications = sum(1 for a, b in zip(reference_dna, fake_dna) if a != b)
    mod_rate = (modifications / len(reference_dna)) * 100

    print(f"\n[4] Statistics:")
    print(f"Number of modifications: {modifications}")
    print(f"Modification rate: {mod_rate:.2f}%")
    print(f"Expansion rate: 0% (no expansion)")
    print(f"bpn (bits per nucleotide): {len(secret_binary) / len(reference_dna):.4f}")

    # Extract and verify
    print(f"\n[5] Extracting hidden Data for verification...")
    extracted_binary = stego.extract_data(fake_dna, reference_dna, num_segments, version)

    print(f"Extracted binary length: {len(extracted_binary)} bits")
    print(f"First 80 bits: {extracted_binary[:80]}...")

    # Verify
    if extracted_binary == secret_binary:
        print("\n✓ SUCCESS: Message extracted correctly!")
    else:
        print("\n✗ ERROR: Extraction failed!")
        # Show first difference
        for i, (a, b) in enumerate(zip(secret_binary, extracted_binary)):
            if a != b:
                print(f"First difference at bit {i}: expected '{a}', got '{b}'")
                break

    # Save outputs
    print(f"\n[6] Saving output files...")

    output_dir = os.path.join(os.path.dirname(__file__), "output", f"RBS_{version}")

    os.makedirs(output_dir, exist_ok=True)

    # Save stego DNA
    stego_file = os.path.join(output_dir, f"{version} stego_DNA.txt")

    with open(stego_file, 'w') as f:
        f.write("# Stego DNA Sequence (with hidden Data)\n")
        f.write(f"# Original DNA file: {dna_file}\n")
        f.write(f"# Secret file: {secret_file}\n")
        f.write(f"# Version: {version}\n")
        f.write(f"# Number of segments: {num_segments}\n")
        f.write(f"# Hidden bits: {len(secret_binary)}\n")
        f.write(f"# Modification rate: {mod_rate:.2f}%\n\n")
        f.write(fake_dna)

    # Save comparison
    compare_file = os.path.join(output_dir, "comparison.txt")
    with open(compare_file, 'w') as f:
        f.write("# DNA Steganography Comparison\n")
        f.write("# Positions marked with '*' indicate modifications\n\n")
        f.write("Position | Reference | Stego | Modified\n")
        f.write("-" * 45 + "\n")
        for i, (orig, fake) in enumerate(zip(reference_dna, fake_dna)):
            if orig != fake:
                f.write(f"{i:8d} | {orig:^9s} | {fake:^5s} | *\n")

    # Save metadata
    metadata_file = os.path.join(output_dir, "metadata.txt")
    with open(metadata_file, 'w') as f:
        f.write("# DNA Steganography Metadata\n\n")
        f.write(f"Original DNA file: {dna_file}\n")
        f.write(f"Secret message file: {secret_file}\n")
        f.write(f"Version: {version}\n")
        f.write(f"Reference DNA length: {len(reference_dna)}\n")
        f.write(f"Stego DNA length: {len(fake_dna)}\n")
        f.write(f"Secret message length: {len(secret_binary)} bits\n")
        f.write(f"Number of segments: {num_segments}\n")
        f.write(f"Modifications: {modifications}\n")
        f.write(f"Modification rate: {mod_rate:.2f}%\n")
        f.write(f"Expansion rate: 0%\n")
        f.write(f"bpn (bits per nucleotide): {len(secret_binary) / len(reference_dna):.4f}\n")
        f.write(f"\nComplementary rule used:\n")
        f.write(f"C(A) = T\n")
        f.write(f"C(C) = A\n")
        f.write(f"C(G) = C\n")
        f.write(f"C(T) = G\n")

    print(f"Files saved to {output_dir}:")
    print(f"  - basic stego_DNA.txt (DNA sequence with hidden Data)")
    print(f"  - comparison.txt (modification details)")
    print(f"  - metadata.txt (embedding statistics)")

    print("\n" + "=" * 70)
    print("Process completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()