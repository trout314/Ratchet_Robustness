venv:
	python3 -m venv ./venv
	./venv/bin/pip3 install -r requirements.txt

clean:
	rm -rf ./venv