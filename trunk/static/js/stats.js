// Pie Chart
/*
Highcharts.chart('container', {
    chart: {
        type: 'pie',
        options3d: {
            enabled: true,
            alpha: 45,
            beta: 0
        }
    },
    title: {
        text: 'Gender'
    },
    tooltip: {
        pointFormat: '{series.name}: <b>{point.percentage:.1f}%</b>'
    },
    plotOptions: {
        pie: {
            allowPointSelect: true,
            cursor: 'pointer',
            depth: 35,
            dataLabels: {
                enabled: true,
                format: '{point.name}'
            }
        }
    },
    series: [{
        type: 'pie',
        name: 'Browser share',
        data: [
            ['Firefox', 45.0],
            ['IE', 26.8],
            ['Chrome', 12.8],
            ['Safari', 8.5],
            ['Opera', 6.2],
            ['Others', 0.7]
        ]
    }]
});
*/

$.chartjq = function(id){
  //alert(id);
	var newDiv = $('<div id="' + id + '" style="height: 400px" class="w3-card-4 col-lg-5 col-md-5 col-sm-12 mx-auto margin-top64"></div>');
  //newDiv.style.background = "#000";
  $('#visualsboard').append(newDiv);
};

function changeGraph(self){
	alert(self.value);
}

function box_checked(self, checkbox)
{
  if (self.checked){
      // alert('checked');
      $.chartjq(checkbox.value);
  }else {
		var parent = document.getElementById("visualsboard");
		var child = document.getElementById(checkbox.value);
		parent.removeChild(child);
  }

  addChart(checkbox.value, checkbox.label, checkbox.stats);

  /*
  if (document.getElementById('xxx').checked)
  {
      document.getElementById('totalCost').value = 10;
  } else {
      calculate();
  }
  */

}

function addChart(gtitle, sname, sdata)
{
  // alert(gtitle);
  var chart = new Highcharts.chart(gtitle, {
      chart: {
          type: 'pie',
          options3d: {
              enabled: true,
              alpha: 45,
              beta: 0
          }
      },
      title: {
          text: gtitle
      },
      tooltip: {
          pointFormat: '{series.name}: <b>{point.percentage:.1f}%</b>'
      },
      plotOptions: {
          pie: {
              allowPointSelect: true,
              cursor: 'pointer',
              depth: 35,
              dataLabels: {
                  enabled: true,
                  format: '{point.name}'
              }
          }
      },
      series: [{
          type: 'pie',
          name: sname,
          data: sdata
      }]
  });

  chart.redraw()
}
