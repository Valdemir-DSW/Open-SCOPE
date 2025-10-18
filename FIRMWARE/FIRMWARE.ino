const int ch1Pin = A0;  // Canal 1
const int ch2Pin = A1;  // Canal 2

void setup() {
  Serial.begin(115200); 
}

void loop() {
  uint16_t ch1 = analogRead(ch1Pin); // 0-1023
  uint16_t ch2 = analogRead(ch2Pin);

  // Enviar em binário (little-endian)
  Serial.write((uint8_t*)&ch1, 2);
  Serial.write((uint8_t*)&ch2, 2);

  delay(0.50); //20000 Amostras
}
