# OpenScope — Professional Client v7

Cliente PyQt5/PyQtGraph para o firmware SCP Scope Direct Stream (STM32F103 e Arduino Leonardo).

## Correções principais desta revisão

Esta versão mantém o layout HD atual e altera o pipeline interno de aquisição/renderização.

### 1. Serial sem fila infinita de sinais Qt

O `SerialWorker` decodifica SCP1 numa thread e coloca os pacotes numa fila thread-safe. A GUI busca os pacotes em lotes a cada 5 ms. Isso evita acumular centenas de `pyqtSignal` na fila do Qt quando o USB CDC entrega dados em rajadas.

A leitura serial é não bloqueante (`timeout=0`) com espera ociosa de 1 ms, em vez de polling bloqueante de 30 ms.

### 2. Gráfico independente da chegada dos pacotes

O gráfico atualiza em timer próprio de aproximadamente 60 Hz. O ritmo do USB não controla mais diretamente o ritmo do desenho.

Aquisição, trigger e desenho agora são processos independentes:

`USB -> decoder -> PC history -> PC trigger -> display timer`

### 3. Pacotes são processados em lote

O cliente não atualiza widgets, medições e gráficos para cada pequeno pacote individual. Um lote inteiro é incorporado ao histórico e só depois a interface é atualizada.

### 4. Gap não apaga mais o histórico

Uma perda de sequência/packet gap não limpa mais todo o ring buffer. O cliente mantém o histórico e apenas interrompe a continuidade do detector de trigger naquele ponto. Isso evita a forma de onda voltar repetidamente para um pequeno trecho no canto da tela.

### 5. Trigger periódico corrigido

O detector agora coleta **todas** as bordas encontradas. O renderer escolhe o trigger mais recente que já possui pós-trigger completo.

Na implementação anterior, o cliente encontrava uma borda, esperava todo o pós-trigger, desenhava e só então começava a procurar outra. Com 100 ms/div e 50% de pre-trigger isso criava aproximadamente 500 ms de atraso por ciclo.

Agora, após o preenchimento inicial, sinais periódicos podem atualizar na cadência do display.

- Trigger OFF: rolling contínuo.
- Trigger Normal: preview até a primeira borda válida; depois mantém a última aquisição estabilizada até existir uma nova pronta.
- Trigger Auto: igual ao Normal, mas força uma aquisição após o timeout quando não encontra uma borda.
- ARM: rearma somente o trigger do PC; não reinicia o firmware.
- FORCE: força trigger usando o histórico já recebido.

### 6. Time/div é somente zoom

Alterar `Time/div` não muda mais automaticamente a taxa do ADC e não reinicia o fluxo.

Isso é proposital. O usuário pode aproximar/afastar a forma de onda instantaneamente usando os dados já existentes no histórico.

A resolução física continua definida pelo perfil/taxa de aquisição. Para ganhar resolução temporal real, selecione `High speed` ou uma taxa manual maior.

### 7. Preenchimento inicial

Enquanto ainda não existe uma tela inteira de histórico, a forma de onda cresce da esquerda para a direita. Ela não fica mais como um pequeno segmento colado no canto direito.

### 8. Persistência mais leve

O traço principal pode atualizar a ~60 Hz, mas o histórico visual de persistência só é atualizado a no máximo 5 Hz no modo rolling. Isso reduz cópias e chamadas `setData()` desnecessárias.

## Firmware

O firmware Direct Stream v6 não precisou ser alterado para estas correções. Ele já opera no modelo correto: ADC -> pequeno bloco de transporte -> USB. Trigger e histórico são do Windows.

## Como abrir

Abra `main.py` com Python para desenvolvimento. Para gerar a distribuição Windows e o instalador, execute `build.bat`.

Para instalar/atualizar apenas as bibliotecas de execução, `run_windows.bat` continua disponível. Também funciona pelo terminal com:

```bash
py -3 main.py
```

## Teste

```bat
py -3 main.py
```

Depois, para um teste simples:

1. Clique `Demo`.
2. Trigger OFF.
3. Selecione `100 ms/div` para visualizar 1 s.
4. Use `-` no Time/div para aproximar sem reiniciar a aquisição.
5. Ative o trigger e use Normal ou Auto.

O self-test pode ser executado com:

```bash
python selftest.py
```

Ele valida SCP1, stream STM32/Leonardo, ring buffer de 1 segundo e detecção de múltiplas bordas.


## Build para Windows

`build.bat` prepara o OpenScope com Nuitka em modo standalone e, quando encontra o Inno Setup, compila também `installer\OpenScope.iss`. A busca do `ISCC.exe` não depende apenas do PATH: verifica locais comuns de instalação, registro do Windows e faz busca de fallback em Program Files.

O ícone `resources\OpenScope.ico` já acompanha o projeto e é usado pelo aplicativo, pelo executável Nuitka e pelo instalador. A área de logo do gráfico continua preparada para `resources\OpenScope_logo.png`; enquanto esse arquivo não existir, nenhum logotipo é desenhado ou exportado.


## Ferramenta adicional

O menu **Ferramentas** inclui a **Calculadora RPM ↔ frequência**, que converte RPM, frequência de dentes, período entre dentes, ângulo por posição e período por volta. Para rodas com falha, informe a quantidade teórica de posições (por exemplo, 36 para uma roda 36-1).
