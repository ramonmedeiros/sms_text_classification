default: run

run:
	[ -f train-data.tsv ] || curl -O https://cdn.freecodecamp.org/project-data/sms/train-data.tsv
	[ -f valid-data.tsv ] || curl -O https://cdn.freecodecamp.org/project-data/sms/valid-data.tsv
	python hello.py

convert:
	tensorflowjs_converter \
    --input_format=tf_saved_model \
    --saved_model_tags=serve \
    saved_model/model \
    web_model
