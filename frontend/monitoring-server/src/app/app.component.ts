import { Component, OnInit, OnDestroy } from '@angular/core';
import { Observable, Subscription } from 'rxjs';

import {
  IMqttMessage,
  MqttService,
} from 'ngx-mqtt';

export interface RegisteredDevice {
  name: string;
  activeDetections: number;
  detectionImg?: string;
  backgroundColor: string;
}

export interface DetectionHistory {
    name: string;
    lastSeen: string;
    travelDirection: string;
    lastSeenTime: Date;
}

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit, OnDestroy {
  private subscription: Subscription;

  title = 'monitoring-server';
  devices: RegisteredDevice[] = [
    {
      name: 'Raspberry_Pi_1',
      activeDetections: 0,
      backgroundColor: 'white', 
    },
    {
      name: 'Raspberry_Pi_2',
      activeDetections: 0, 
      backgroundColor: 'white', 
    },
    {
      name: 'Raspberry_Pi_3',
      activeDetections: 0, 
      backgroundColor: 'white', 
    }
  ];

  detectionHistory: DetectionHistory[] = [
    {
      name: 'Tiger_0',
      lastSeen: 'Raspberry_Pi_2',
      travelDirection: 'Left, Down, and inwards',
      lastSeenTime: new Date('2/17/20'),
    },
    {
      name: 'Tiger_2',
      lastSeen: 'Raspberry_Pi_3',
      travelDirection: 'Left, Still, and Still',
      lastSeenTime: new Date('2/17/20'),
    },  
  ]; 

  regChannels = new Map<string, Subscription>();

  addDevice(message: IMqttMessage): void {
    let msg = message.payload.toString();
    this.devices.push( { name: msg, activeDetections: 0, backgroundColor: 'white' });

    // Add a subscription for device
    if (this.regChannels.has(msg)) {
      console.log('Device ' + msg + ' already registered!');
    } else {
      console.log('Subscribing to ' + msg);
      this.regChannels.set(msg, this._mqttService.observe(msg).subscribe(this.detectionReceived.bind(this)));
    }
  }

  detectionReceived(message: IMqttMessage): void {
    let channel = message.topic.toString();
    let msg = message.payload.toString();
    console.log('Detection received on ' + channel + ', : ' + msg);
  }

  constructor(private _mqttService: MqttService) {
    this.subscription = this._mqttService.observe('registration').subscribe(this.addDevice.bind(this));
    console.log('Subscribed');
  }

  setImages(): void {
      this.devices.forEach((device) => { 
          device.detectionImg = "https://material.angular.io/assets/img/examples/shiba2.jpg";
      });
  }

  ngOnInit() :void {
      var tmp = this;
      setTimeout(function() {
          tmp.devices.forEach((device) => { 
              device.detectionImg = "https://material.angular.io/assets/img/examples/shiba2.jpg";
          });
      }, 10000);

      setTimeout(function () {
          tmp.devices.push({ name: 'Raspberry_Pi_4', activeDetections: 0, backgroundColor: 'white',});
          tmp.devices[0].backgroundColor = 'red';
      }, 15000);
  }

  ngOnDestroy(): void {
    this.regChannels.forEach((value, key) => {
      value.unsubscribe();
    });
    this.subscription.unsubscribe();
  }

}
