$(document).ready(function() {
  $('table.display').DataTable({
    "scrollY": "400px",
    "paging": false,
    "responsive": true
  });


  /*
  var table = $('#important_features').DataTable({
    "scrollY": "400px",
    "paging": false,
    "responsive": true
  });

  var table = $('#kmeans').DataTable({
    "scrollY": "400px",
    "paging": false,
    "responsive": true
  });

  var table = $('#svm').DataTable({
    "scrollY": "400px",
    "paging": false,
    "responsive": true
  });

  var table = $('#keras').DataTable({
    "scrollY": "400px",
    "paging": false,
    "responsive": true
  });

  var table = $('#knn').DataTable({
    "scrollY": "400px",
    "paging": false,
    "responsive": true
  });
  */



  /*
    $('#important_features tbody').on('click', 'tr', function () {
        var data = table.row( this ).data();
        alert( 'You clicked on row' + data);
    } );
    */
} );
