# Generated by Django 3.1 on 2020-08-21 14:14

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('flower', '0002_image_first_name'),
    ]

    operations = [
        migrations.AlterField(
            model_name='image',
            name='first_name',
            field=models.CharField(default='', max_length=20),
        ),
    ]
