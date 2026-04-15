# minips

A parameter server for distributed training, written in Go with PyTorch workers. The server shards model weights across goroutines and applies SGD with momentum as gradients arrive over gRPC. Workers synchronise under bulk-synchronous, bounded-staleness or fully asynchronous rules, and can compress their gradients with top-k sparsification (with error feedback) or 8-bit quantisation. A torch.distributed all-reduce baseline trains the same model on the same data for comparison.
