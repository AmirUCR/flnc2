#!/usr/bin/env python
#encoding=utf-8
import os
import re
import sys
import operator
#import multiprocessing
#from Bio import SeqIO
import math
import random
import optparse
import time
import subprocess
num_str=''
reverse = False
species = ''
Start_Codon = 0
Genetic_Code = 1

ALLOWED_SPECIES = [
    "Homo_sapiens", 
    "Mus_musculus", 
    "Danio_rerio", 
    "Caenorhabditis_elegans", 
    "Oryza_sativa", 
    "Solanum_lycopersicum"
]

def check_species(option, opt_str, value, parser):
    if value not in ALLOWED_SPECIES:
        raise optparse.OptionValueError(
            f"Option {opt_str}: invalid choice: {value} (choose from {', '.join(ALLOWED_SPECIES)})"
        )
    setattr(parser.values, option.dest, value)

def desStartCode(codes):
	if(codes in ("ATG","atg")):
		return True
	return False

def desEndCode(codes):
	if(codes in ("TAA","taa","tag","tga", "TAG", "TGA")):
		return True
	return False


def readByThree(string, offset):
    length = len(string)
    result = set()

    i = 0
    while i < length - 2:
        start = -1
        end = -1
        # Find the start codon
        while i < length - 2:
            codes = string[i:i+3]
            if desStartCode(codes):
                start = i
                break
            i += 3

        # If no start codon found, break
        if start == -1:
            break
        
        # Move to the next codon position to start searching for stop codon
        j = start + 3
        while j < length - 2:
            codes = string[j:j+3]
            if desEndCode(codes):
                end = j + 2
                result.add((start + offset, end + offset))
                break
            j += 3
        
        # Continue to find the next ORF starting from the next codon after the previous start codon
        i = start + 3
    
    return result


def getGC(string):
	GC = string.count("G") + string.count("C") + string.count("g") + string.count("c")
	GC2 = GC/(len(string)+0.0)
	return GC2

def getInfo(string, pos):
	length = pos[1] - pos[0] +1
	gc = getGC(string[pos[0]:pos[1]+1])
	return str(pos[0]), str(pos[1]), str(length), str(gc)

def writeInfo(resultInfo,name,num_str):
	nf = open('ORF'+num_str+'.temp', "a")
	for r in resultInfo:
		nf.write("\t".join(r) + "\n")
	a = name.strip().split()
	nf.write(a[0] + "\n")
	nf.close()

def length_GC(string):
	G = string.count("G") + string.count("g")
	C = string.count("C") + string.count("c")
	L = len(string)
	return str(L) + '\t' + str(G) + '\t' + str(C) + '\n'

def reverse_complement(dna_sequence):
    # Define the complement for each nucleotide
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'a': 't', 't': 'a', 'c': 'g', 'g': 'c'}
    
    # Generate the complement sequence
    complement_sequence = ''.join(complement[nucleotide] for nucleotide in dna_sequence)
    
    # Reverse the complement sequence
    reverse_complement_sequence = complement_sequence[::-1]
    
    return reverse_complement_sequence


def process_strings(lines, resultInfo):
    strings = [lines, lines[1:], lines[2:]]
    for index, string in enumerate(strings):
        positions = readByThree(string, index)
        positions = sorted(positions, key=operator.itemgetter(0))
        for pos in positions:
            resultInfo.append(getInfo(lines, pos))

def process_sequence(lines, name1, num_str, R):
    resultInfo = []
    if R:
        lines = reverse_complement(lines)
    process_strings(lines, resultInfo)
    writeInfo(resultInfo, name1[0], num_str)

def ORF(f1, num_str):
    def handle_sequences(f, R):
        name = f.readline().split('>')
        name1 = name[1].split(' ')
        lines = ''

        for line in f:
            if line.startswith(">"):
                process_sequence(lines, name1, num_str, R)
                name = line.split('>')
                name1 = name[1].split(' ')
                lines = ''
            else:
                lines += line.strip()

        # Process the last sequence
        process_sequence(lines, name1, num_str, R)

    with open(f1) as f:
        handle_sequences(f, False)
        if reverse:
            f.seek(0)  # Reset file pointer to the beginning
            handle_sequences(f, True)

def run_ORFfinder(input_fasta, output_file, start_codon=Start_Codon, genetic_code=Genetic_Code, min_length=100, strand = "plus"):
    """
    Run ORFfinder on a given input FASTA file.

    :param input_fasta: Path to the input FASTA file
    :param output_file: Path to the output file
    :param start_codon: Start codon to be used (default is 1)
    :param min_length: Minimum ORF length to be considered (default is 100)
    """
    try:
        if reverse == True:
            strand = "both"
        # Construct the ORFfinder command
        command = [
            # "./ORFfinder",
			orffinder_path,
            "-in", input_fasta,
            "-out", output_file,
            "-s", str(start_codon),
            "-g", str(genetic_code),
            "-ml", str(min_length),
            "-outfmt", str(1),
            "-strand", strand
        ]
        
	# Open a null file to suppress error messages
        with open(os.devnull, 'w') as devnull:
            # Execute the command, redirecting stderr to devnull to hide errors
            subprocess.run(command, check=True, stderr=devnull)
        
        print(f"ORFfinder analysis completed.")
    
    except subprocess.CalledProcessError as e:
        print(f"Error running ORFfinder: {e}")

def LenTop3(path,opath,top):
	f = open(path,'r')
	nf = open(opath,'a')
	length = []
	for line in f:
		a = line.strip()
		a = a.split('\t')
		if len(a) == 4:
			a[2] = int(a[2])
			length.append(a)
			continue
		else:
			if len(length) != 0:
				length.sort(key=operator.itemgetter(2), reverse = True)
				for i in range(0,min(top,len(length))):
					length[i][2] = str(length[i][2])
					if int(length[i][2])>100:
						nf.write('\t'.join(length[i]) + '\n')
			nf.write(line.strip() + '>\n')
			length = []
	f.close()
	nf.close()

def parse_fasta(fasta_file):
    with open(fasta_file, 'r') as file:
        sequences = {}
        seq_id = None
        sequence = ''
        header_info = None
        #print("parse_fasta starts")

        for line in file:
            line = line.strip()
            if line.startswith('>'):
                if seq_id:
                    if seq_id not in sequences:
                        sequences[seq_id] = []
                    sequences[seq_id].append((header_info, sequence))
                    sequence = ''
                header_info = line
                #match = re.search(r"(OS\d{2}T\d{7}-\d{2}_cdna)", header_info)
                #if match:
                #     seq_id = match.group(1)
                name = line.split(">")
                name1 = name[1].split(" ")
                name2 = name1[0].split(":")
                seq_id = name2[0]
            else:
                sequence += line
        if seq_id:
            if seq_id not in sequences:
                sequences[seq_id] = []
            sequences[seq_id].append((header_info, sequence))
        #print("parse_fasta ends")    

    return sequences

def get_orf_details(header_info):
    match = re.search(r':(\d+):(\d+)', header_info)
    if match:
        start = int(match.group(1))
        end = int(match.group(2))
        if end > start:
            length = end - start + 1
            return start, end, length
        else:
            length = start - end + 1
            return end, start, length
    return None, None, 0

def calculate_gc_content(sequence):
    g = sequence.count('G')
    c = sequence.count('C')
    a = sequence.count('A')
    t = sequence.count('T')
    gc_content = (g + c) / (g + c + a + t) if (g + c + a + t) > 0 else 0
    return gc_content

def process_fasta(fasta_file, output_file, id_mapping):
    sequences = parse_fasta(fasta_file)
    seq_ids = list(id_mapping.keys())
    seq_ids_iter = iter(seq_ids)
    seq_id_ordered = next(seq_ids_iter)
    with open(output_file, 'w') as out_file:
        for seq_id, orfs in sequences.items():
            while seq_id != seq_id_ordered:
                out_file.write(f"0\t0\t0\t0\n")
                out_file.write(f"{seq_id_ordered}>\n")
                seq_id_ordered = next(seq_ids_iter)            

            orf_details = []
            for header_info, sequence in orfs:
                start, end, length = get_orf_details(header_info)
                gc_content = calculate_gc_content(sequence)
                orf_details.append((start, end, length, gc_content))
            
            orf_details.sort(key=lambda x: x[2], reverse=True)
            
            """  
            top_orfs = orf_details[:3]
            
            for orf in top_orfs:
                out_file.write(f"{orf[0]}\t{orf[1]}\t{orf[2]}\t{orf[3]}\n")
            out_file.write(f"{seq_id}>\n")
            """
            selected_orfs = []
            for orf in orf_details:
                start, end, _, _ = orf
                overlap = False
                for selected_orf in selected_orfs:
                    selected_start, selected_end, _, _ = selected_orf
                    if not (end <= selected_start or start >= selected_end):
                        overlap = True
                        break
                if not overlap:
                    selected_orfs.append(orf)
                if len(selected_orfs) == 3:
                    break

            for orf in selected_orfs:
                out_file.write(f"{orf[0]}\t{orf[1]}\t{orf[2]}\t{orf[3]}\n")
            out_file.write(f"{seq_id}>\n")

            try:
                # Get the next element from the iterator
                seq_id_ordered = next(seq_ids_iter)
            except StopIteration:
            # Break the loop when the iterator is exhausted
                break
            

def create_temp_fasta(original_fasta, temp_fasta):
    # Dictionary to store the mapping from original identifiers to new IDs
    id_mapping = {}
    # Counter for generating simple IDs
    id_counter = 1
    
    with open(original_fasta, 'r') as infile, open(temp_fasta, 'w') as outfile:
        for line in infile:
            if line.startswith('>'):
                # Extract the original identifier
                original_id = line[1:].strip()
                # Create a new simple ID
                new_id = f"seq{id_counter}"
                # Increment the ID counter
                id_counter += 1
                # Store the mapping
                id_mapping["lcl|"+new_id] = original_id
                # Write the new identifier to the temporary FASTA file
                outfile.write(f">{new_id}\n")
            else:
                # Write the sequence lines unchanged
                outfile.write(line)
    
    return id_mapping


def main(f2, f3):
	#print "Running LGC with %d process(es)" % processes
	print(("Input: " + f2))
	#print "param:" + f4
	print(("Output: " + f3))

	#f=open(f4,'r')
	#s = f.read().split('\n')
	s_All = [[-0.05215,0.10796,-0.071697,0.01759],[0.1014,-0.16872,0.0858349,-0.0083821]]
	s_Homo_sapiens = [[0.0247,-0.1109,0.1781,-0.0933], [0.0018,0.0256,-0.0577,0.0356]]
	#s_Mus_musculus =  [[0.0295,-0.1241,0.1773,-0.0776], [0.0064,-0.0035,0.0024,-0.0043]]
	s_Mus_musculus = [[0.009833, -0.041367, 0.0591, -0.025867], [0.002133, -0.00116667, 0.0008, -0.00143333]]
	s_Danio_rerio =  [[0.0499,-0.2116,0.2772,-0.0941], [0.0200,-0.0705,0.0919,-0.0315]]
	s_Caenorhabditis_elegans = [[0.0707,-0.4205,0.8338,-0.5340],[0.0067,-0.0069,0.0084,-0.0001]]
	s_Oryza_sativa = [[0.0246,-0.1130,0.1803,-0.0933], [0.0177,-0.0773,0.1568,-0.1033]]
	#s_Oryza_sativa = [[0.0177,-0.0773,0.1568,-0.1033], [0.0246,-0.1130,0.1803,0.0933]]
	s_Solanum_lycopersicum = [[0.0649,-0.3437,0.5536,-0.2249],[-0.0478,0.3571,-0.7745,0.5419]]
	#f.close()
	if species == "Homo_sapiens":
		c = s_Homo_sapiens[0]
		n = s_Homo_sapiens[1]
	elif species == "Mus_musculus":
		c = s_Mus_musculus[0]
		n = s_Mus_musculus[1]
	elif species == "Danio_rerio":
		c = s_Danio_rerio[0]
		n = s_Danio_rerio[1]
	elif species == "Caenorhabditis_elegans":
		c = s_Caenorhabditis_elegans[0]
		n = s_Caenorhabditis_elegans[1]
	elif species == "Oryza_sativa":
		c = s_Oryza_sativa[0]
		n = s_Oryza_sativa[1]
	elif species == "Solanum_lycopersicum":
		c = s_Solanum_lycopersicum[0]
		n = s_Solanum_lycopersicum[1]
	else:
		c = s_All[0]
		n = s_All[1]
	#print c
	#print n

	"""
	num_list = ['0','1','2','3','4','5','6','7','8','9']
	random.shuffle(num_list)
	num_str = ''.join(num_list)
	print('Scan ORF ...')
	ORF(f2,num_str)
	print('Done')
	oopath = 'ORF'+num_str+'.temp'
	outpath1 = 'top'+num_str+'.temp'
	"""
	# Generate a unique timestamp
	timestamp = time.strftime("%Y%m%d%H%M%S")
	oopath = f'ORF{timestamp}.temp'

	print('Scan ORF ...')
	#ORF(f2, timestamp)
	f1 = f'simp{timestamp}.temp'
	id_mapping = create_temp_fasta(f2, f1)
	run_ORFfinder(f1, oopath)
	print('Done')
	outpath1 = f'top{timestamp}.temp'
	outpath2 = f3


	process_fasta(oopath,outpath1,id_mapping)
	oo = open(outpath2,'w')
	a1 = c[0]
	a2 = c[1]
	a3 = c[2]
	a4 = c[3]
	b1 = n[0]
	b2 = n[1]
	b3 = n[2]
	b4 = n[3]
	g = open(outpath1,'r')
	para = g.read().split('>\n')
	m = len(para)
	Fc = [1]*m
	Fn = [1]*m
	line = ['',-1,-1,-1,-1]*m
	Jcomp  = [-1]*m
	string = [0]*m
	J      = [0]*m
	Cflag  = [0]*m
	"""
	oo.write("# Sequence Name: name of transcript\n")
	oo.write("# ORF Length: length of the longest ORF\n")
	oo.write("# GC Content: GC content of the longest ORF\n")
	oo.write("# Coding Potential Score: coding potential score for a transcript, which is protein-coding RNA if greater than 0 or ncRNA if smaller than 0. '0' indicates that mRNA probability equals lncRNA probability. Also, if the ORF length is shorter than 100nt, '0' is output.\n")
	oo.write("# pc: ORF probability for coding sequence\n")
	oo.write("# pnc: ORF probability for non-coding sequence\n")
	oo.write("# fc: Stop-codon probability for coding sequence\n")
	oo.write("# fnc: Stop-codon probability for non-coding sequence\n")
	oo.write("# Coding Label: 'Coding' represents mRNA and 'Non-coding' represents ncRNA\n")
	oo.write("# \n")
	oo.write("# Sequence Name\t")
	oo.write("ORF Length\t")
	oo.write("ORF Start\t")
	oo.write("ORF End\t")
	oo.write("GC Content\t")
	oo.write("Coding Potential Score\t")
	oo.write("Coding Label\t")
	oo.write("pc\t")
	oo.write("pnc\t")
	oo.write("fc\t")
	oo.write("fnc\t")
	oo.write("\n")
	"""
	headers = ["# Sequence Name", "ORF Length", "ORF Start", "ORF End", "GC Content", "Coding Potential Score", "Coding Label", "pc", "pnc", "fc", "fnc"]
	data_lines = []
	for i in range(0,m):

		segment = para[i].split('\n')

		if len(segment) == 1:
			J[i] = 1


		number  = segment.pop()
		string[i] = len(segment)
		fc  = [0]*len(segment)
		fn  = [0]*len(segment)
		C   = [0]*len(segment)
		LL_f = 0
		GC_f = 0
		start_f = 0
		end_f = 0

		ffc = 1
		ffn = 1


		for j in range(0,len(segment)):
			seg = segment[j].split('\t')
			C[j]= eval(seg[3])

			GC = eval(seg[3])
			GC2 = GC**2
			GC3 = GC**3

			L  = eval(seg[2])
			N = L/3-1

			pc = 3*(a1*GC3+a2*GC2+a3*GC+a4)
			pn = 3*(b1*GC3+b2*GC2+b3*GC+b4)
			fc[j] = (1-pc)**min(N,5000)*pc
			fn[j] = (1-pn)**min(N,5000)*pn
			if j == 0:
				ffc = pc
				ffn = pn
			if LL_f <= L:
				LL_f = L
				GC_f = GC
				start_f = seg[0]
				end_f = seg[1]

		for j in range(0,len(segment)):
			Fc[i] = Fc[i]*fc[j]
			Fn[i] = Fn[i]*fn[j]
		if Fn[i] < 1e-99:
			Fn[i] = 1e-99
		S = Fc[i]/Fn[i]

		compare = [x<y for x,y in zip(C,[0.345]*len(segment))]
		if False not in compare:
			Cflag[i] = True

		if S >= 1:
			Jcomp[i] = 'Coding'
		else:
			Jcomp[i] = 'Non-coding'

		if Cflag[i] is True:
			Jcomp[i] = 'Non-coding'

		if J[i] == 1:
			Jcomp[i] = 'Non-coding'

		if S <= 0:
			SS = -10
		else:
			SS = math.log(S)

		line[i] = [number,LL_f,GC_f,SS,Jcomp[i],start_f,end_f]
		if len(number) != 0:
		#version 2.0 April 2018 zz
			#sequence name
			#oo.write(line[i][0]+'\t')
			#orf length
			#oo.write(str(line[i][1])+'\t')
			#start
			#oo.write(str(line[i][5])+'\t')
			#end
			#oo.write(str(line[i][6])+'\t')
			
			#GC content
			a_1 = round(line[i][2],3)
			if a_1==0:
				a_1 = format(line[i][2],'.3e')
			#oo.write(str(a_1)+'\t')
			
			#coding potential score
			a_2 = round(line[i][3],3)
			if a_2==0:
				a_2 = format(line[i][3],'.3e')
			#oo.write(str(a_2)+'\t')
			
			#coding label
			#oo.write(str(line[i][4])+'\t')
			
			#fc
			#oo.write(str(format(Fc[i],'.3e'))+'\t')
			
			#fnc
			#oo.write(str(format(Fn[i],'.3e'))+'\t')
			
			#pc
			a_3 = round(ffc,3)
			if a_3==0:
				a_3 = format(ffc,'.3e')
			#oo.write(str(a_3)+'\t')
			
			#pnc
			a_4 = round(ffn,3)
			if a_4==0:
				a_4 = format(ffn,'.3e')
			#oo.write(str(a_4)+'\t')
			
			data_lines.append([id_mapping[line[i][0]],str(line[i][1]),str(line[i][5]),str(line[i][6]),str(a_1),str(a_2),str(line[i][4]),str(format(Fc[i],'.3e')),str(format(Fn[i],'.3e')),str(a_3),str(a_4)])

	Chow = Cflag.count(True)
	
	headers_row = '\t'.join(headers)
	oo.write(headers_row + '\n')
	for row in data_lines:
		oo.write('\t'.join(row) + '\n')

	'''
	# Calculate the max length of each column
	max_lengths = [
		max(len(header), *(len(str(row[i])) for row in data_lines)) for i, header in enumerate(headers)
	]

	# Format the header to be right-aligned based on the maximum length
	formatted_headers = [header.rjust(max_length) for header, max_length in zip(headers, max_lengths)]
	formatted_header_row = '   '.join(formatted_headers)

	# Format each data line to be right-aligned based on the maximum length
	formatted_data_rows = []
	for data in data_lines:
		formatted_data = [value.rjust(max_length) for value, max_length in zip(data, max_lengths)]
		formatted_data_row = '   '.join(formatted_data)
		formatted_data_rows.append(formatted_data_row)

	# Write the formatted content to a file
	oo.write(formatted_header_row + '\n')
	for row in formatted_data_rows:
		oo.write(row + '\n')
	'''

	oo.close()
	g.close()
	os.remove(oopath)
	os.remove(outpath1)


if __name__ == '__main__':
	from optparse import OptionParser
	usage = "LGC version 1.0\nusage: lgc.py [options] INPUT OUTPUT\nPlease cite: \"LGC: Characterization and Identification of Long Non-coding RNAs Based on Feature Relationship\""
	parser = OptionParser(usage=usage)

	parser.add_option("-p", "--processes", dest="processes", type="int",
	                  help="Processes to run LGC", metavar="PROCESSES", default=1)
	parser.add_option("-r", "--reverse", dest="reverse", action="store_true",
				   	  help="Check the reverse strand", metavar="REVERSE", default=False)
	parser.add_option("-S", "--species", dest="species", type="string",
                      help="Species to use in the process", metavar="SPECIES",
                      action="callback", callback=check_species)
	parser.add_option("-s", "--start_codon", dest="start_codon", type="int",
					  help="ORF start codon to use", metavar="INI", default=1)
	parser.add_option("-g", "--genetic_code", dest="genetic_code", type="int",
					  help="Genetic code to use", metavar="GENCODE", default=0)
	
	# Patched by Amir
	parser.add_option("--orffinder_path", dest="orffinder_path", type="string")

	(options, args) = parser.parse_args()
	#print len(args)
	if len(args) != 2:
		print(usage)
		exit()
	input = args[0]
	#print input
	#path = args[0]
	#print path
	output = args[1]
	#print output
	import time
	start_time = time.time()
	#print start_time
	reverse = options.reverse
	species = options.species
	Start_Codon = options.start_codon
	Genetic_Code = options.genetic_code
	orffinder_path = options.orffinder_path
	main(input, output)
	print("Computation time %s senconds" % (time.time() - start_time))
