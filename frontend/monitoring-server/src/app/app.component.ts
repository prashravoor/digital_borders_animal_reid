import { Component, OnInit } from '@angular/core';

export interface RegisteredDevice {
  name: string;
  activeDetections: number;
  detectionImg?: string;
}

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit {
  title = 'monitoring-server';
    devices: RegisteredDevice[] = [
    {
      name: 'Raspberry_Pi_1',
      activeDetections: 0, 
    },
    {
      name: 'Raspberry_Pi_2',
      activeDetections: 0, 
    },
    {
      name: 'Raspberry_Pi_3',
      activeDetections: 0, 
    }
  ];

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
          tmp.devices.push({ name: 'Raspberry_Pi_4', activeDetections: 0});
      }, 15000);
  }
}
