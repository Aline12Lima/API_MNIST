# API MNIST

API FastAPI para classificar imagens de dígitos usando os pesos treinados no Kaggle.

## Executar

Na pasta do projeto:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
uvicorn main:app --reload
```

A instalação acima usa PyTorch para CPU, como o carregador do modelo.
O arquivo `MulticlassCNN.pth` deve ficar ao lado de `model.py`.
O modelo é carregado uma única vez por processo na inicialização da API.

Abra http://127.0.0.1:8000 para usar a interface: desenhe um dígito com o mouse
ou toque, ou envie uma imagem PNG, JPG ou WEBP de até 5 MB. Para imagens com
fundo claro, marque a opção de inverter cores. Clique em **Reconhecer dígito**
para ver a previsão e a confiança.

A documentação continua em http://127.0.0.1:8000/docs. Use `POST /predict` para enviar uma imagem
no campo `file`, ou execute:

```bash
curl -X POST http://127.0.0.1:8000/predict -F "file=@digito.png"
```

A resposta contém `numero_previsto` (0 a 9) e `confianca` (softmax entre 0 e 1).
`GET /` abre a interface e `GET /health` confirma que a API está funcionando. Arquivos inválidos retornam 400;
requisições sem o campo `file` retornam 422.

O pré-processamento segue o código fornecido: escala de cinza, tamanho 28×28
e tensor com valores de 0 a 1. Use um dígito claro centralizado em fundo preto,
como no MNIST. Caso o treinamento tenha usado normalização adicional, replique-a
em `main.py` antes de avaliar a precisão. A confiança não garante acerto.

Referências: [carregamento no PyTorch](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
e [ciclo de vida do FastAPI](https://fastapi.tiangolo.com/advanced/events/).
