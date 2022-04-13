from argparse import ArgumentParser

parser = ArgumentParser(
    prog='add_eos-to-arpa.py',
    description='Add end-of-sentence token to ARPA file built using KenLM',
)

parser.add_argument('input_arpa',  help = "input file to modify")
parser.add_argument('output_arpa',  help = "output file to write to")

args = parser.parse_args()

# Function adapted from https://huggingface.co/blog/wav2vec2-with-ngram
# Originally written by Patrick von Platen, 2022
with open(args.input_arpa, "r") as read_file, open(args.output_arpa, "w") as write_file:
    has_added_eos = False
    
    for line in read_file:
        if not has_added_eos and "ngram 1=" in line:
            count=line.strip().split("=")[-1]
            write_file.write(line.replace(f"{count}", f"{int(count)+1}"))
            
        elif not has_added_eos and "<s>" in line:
            write_file.write(line)
            write_file.write(line.replace("<s>", "</s>"))
            has_added_eos = True
        
        else:
            write_file.write(line)
