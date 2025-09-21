.PHONY: features train full clean

features:
python -m src.build_features --config config.yaml

train:
python -m src.train_eval

# Full run: edit config.yaml to set subset.enable: false, then:
full: features train

clean:
rm -f data/processed/*.csv reports/*.png reports/*.json
