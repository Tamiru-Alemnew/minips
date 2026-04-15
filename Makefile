PY ?= python3
GO_PKG := github.com/Tamiru-Alemnew/minips/server/internal/pb

.PHONY: build proto test test-go test-py integration lint bench reproduce smoke plot clean

build:
	cd server && go build -o ../bin/ps-server ./cmd/ps-server

proto:
	$(PY) -m grpc_tools.protoc -Iproto --go_out=server/internal/pb --go_opt=module=$(GO_PKG) \
		--go-grpc_out=server/internal/pb --go-grpc_opt=module=$(GO_PKG) proto/ps.proto
	$(PY) -m grpc_tools.protoc -Iproto --python_out=worker/minips/pb --grpc_python_out=worker/minips/pb proto/ps.proto
	sed -i.bak 's/^import ps_pb2/from . import ps_pb2/' worker/minips/pb/ps_pb2_grpc.py && rm worker/minips/pb/ps_pb2_grpc.py.bak

test: test-go test-py

test-go:
	cd server && go vet ./... && go test -race ./...

test-py: build
	PYTHONPATH=worker $(PY) -m pytest -q tests

lint:
	cd server && test -z "$$(gofmt -l .)" && go vet ./...
	$(PY) -m ruff check worker experiments baselines tests

bench: build
	$(PY) experiments/sweep.py --matrix smoke

reproduce: build
	$(PY) experiments/sweep.py --matrix full
	$(PY) experiments/plot.py

smoke: build
	$(PY) experiments/sweep.py --matrix smoke
	$(PY) experiments/plot.py --matrix smoke

plot:
	$(PY) experiments/plot.py

clean:
	rm -rf bin runs
