#include <SoftwareSerial.h>

SoftwareSerial EspSerial(3, 1); // RX TX
String receivedString = "";

void setup() {
  pinMode(3, INPUT);
  pinMode(1, OUTPUT);
  EspSerial.begin(9600);

  Serial.begin(9600);
}

void loop() {
  while(EspSerial.available() > 0) {
    receivedString = EspSerial.readStringUntil('\n');
    Serial.println(receivedString);
  }

  // Your other code or actions can go here

  delay(100);
}