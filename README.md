# Detecção de Movimento com OpenCV

[Be sure to checkout our wiki!](https://github.com/devluisrodrigues/motion-detection/wiki)

Este projeto implementa um sistema de detecção de movimento usando a biblioteca OpenCV em Python. Quando um movimento é detectado, o sistema grava um vídeo de 17 segundos.

## Exemplo de funcionamneto:

Os vídeos abaixo mostram demonstram o funcionamento do programa com pássaros:

![](https://github.com/devluisrodrigues/motion-detection/blob/main/videos_exemplo/passaro.gif)

![](https://github.com/devluisrodrigues/motion-detection/blob/main/videos_exemplo/passaro2.gif)

## Requisitos

python >= 3.9

opencv-contrib-python==4.10.0.84

opencv-python==4.10.0.84

Consulte o arquivo requirements.txt

## Instalação

1. Clone o repositório:

```git clone https://github.com/devluisrodrigues/motion-detection```
```cd motion-detection```

2. Instale as dependências:

```pip install -r requirements.txt```

## Uso

1. Execute o script motion_detection.py:

```python motion_detection.py```

2. O script abrirá a câmera padrão do seu dispositivo e começará a monitorar o movimento. Quando um movimento for detectado, um vídeo será gravado e salvo no diretório atual.

## Parâmetros

Você pode ajustar os seguintes parâmetros no script motion_detection.py:

- MIN_CONTOUR_AREA: Área mínima do contorno para considerar como movimento. O valor padrão é 15.
- VIDEO_DURATION: Tempo de vídeo que será gravado após detectar o movimento. O valor padrão é 17 segundos.

## Funcionamento

1. O script captura frames da câmera em tempo real.
2. Converte os frames para escala de cinza e aplica um desfoque gaussiano.
3. Compara o frame atual com o frame anterior para detectar mudanças.
4. Se a mudança for significativa (baseada na área do contorno), um vídeo é gravado.
5. O vídeo é salvo no formato AVI com um timestamp no nome do arquivo.

## Contribuição

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou enviar pull requests.

## Licença

Este projeto está licenciado sob a Licença MIT. Veja o arquivo LICENSE para mais detalhes.
