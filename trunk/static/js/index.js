$(document).ready(function() {


  var table = $('#important_features').DataTable({
    "scrollY": "400px",
    "paging": false,
    "responsive": true
  });

  var table = $('#select_visual').DataTable({
    "scrollY": "400px",
    "paging": false,
    "responsive": true
  });

  /*
    $('#important_features tbody').on('click', 'tr', function () {
        var data = table.row( this ).data();
        alert( 'You clicked on row' + data);
    } );
    */
} );
