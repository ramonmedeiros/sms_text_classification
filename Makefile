default: run

run:
	[ -f train-data.tsv ] || curl -O https://cdn.freecodecamp.org/project-data/sms/train-data.tsv
	[ -f valid-data.tsv ] || curl -O https://cdn.freecodecamp.org/project-data/sms/valid-data.tsv
	python hello.py
