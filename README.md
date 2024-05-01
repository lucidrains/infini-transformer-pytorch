<img src="./infini-attention.png" width="300px"></img>

## Infini-Transformer - Pytorch (wip)

Implementation of <a href="https://arxiv.org/abs/2404.07143">Infini-Transformer</a> in Pytorch. They use a linear attention scheme to compress past memories and demonstrate multiple SOTAs for long context benchmarks.

Although unlikely to beat <a href="https://github.com/lucidrains/ring-attention-pytorch">Ring Attention</a>, I think it is worth exploring, as the techniques are orthogonal.

## Citations

```bibtex
@inproceedings{Munkhdalai2024LeaveNC,
    title   = {Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention},
    author  = {Tsendsuren Munkhdalai and Manaal Faruqui and Siddharth Gopal},
    year    = {2024},
    url     = {https://api.semanticscholar.org/CorpusID:269033427}
}
```
