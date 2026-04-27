## Week 1
- instalação do ambiente no Mac
- criação da virtual environment
- instalação das dependências do projeto
- configuração do Webots para usar o Python da venv
- clone do repositório
- abertura do world original `smart-wheelchairs.wbt`
- primeiros testes ao arranque dos controladores
- ajustes para smoke test mais leve
- correção de paths e pequenos bugs no arranque
- alinhamento entre número de robôs no world e no treino
- primeiro treino completo
- primeiro teste completo
- geração de modelo e ficheiros de output

## Week 2
- organização da baseline
- criação de notes e artifacts
- adição de métricas no `info` do ambiente
- alteração do `rl-test.py` para guardar métricas por episódio
- criação de `episode_metrics.csv`
- baseline com treino mais longo
- comparação inicial de resultados
- tornar `WheelchairEnv` configurável para `lidar_dim`
- downsampling do LiDAR no ambiente
- tornar a CNN compatível com diferentes números de raios
- adaptação do treino e do teste para usar `LIDAR_DIM`
- experiências com 360, 90 e 45 raios
- comparação entre essas configurações
- treino mais longo com 360 raios
- correção do uso de `VecNormalize` entre treino e teste
- guardar e carregar ficheiro `.pkl` do `VecNormalize`
- teste de reward simplificado
- investigação de problemas no reset
- estabilização do teste final

## Checks
- projeto a correr localmente
- treino funcional
- teste funcional
- baseline documentada
- métricas por episódio
- experiências iniciais com número de raios
- artefactos guardados
